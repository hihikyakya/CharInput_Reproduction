"""Train fixed-target causal bigram probes and write per-seed results."""
import os

# Set this before importing NumPy. Single-thread BLAS also limits run time on laptops.
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np

ROOT = Path(__file__).resolve().parent
CORPUS_HASH = "1afac2fbe9bfb76f5eb8eff6c888f13378bb226c3859c4fa3c416532af62018f"
SEEDS = (11, 23, 37, 53, 71)
VOCAB_SIZE = 512
LENGTH = 16
CHANNELS = 8
UPDATES = 120
RATE = 0.015
SPLITS = ("train", "val", "test")


def load_data():
    raw = gzip.decompress((ROOT / "data/probe.jsonl.gz").read_bytes())
    if hashlib.sha256(raw).hexdigest() != CORPUS_HASH:
        raise ValueError("Corpus checksum differs from the bundled reference corpus.")
    rows = [json.loads(line) for line in raw.splitlines()]
    # No exact duplicate paragraph or overlap in topic names between splits.
    assert len({tuple(row["words"]) for row in rows}) == len(rows)
    topics = [{r["topic"] for r in rows if r["split"] == s} for s in SPLITS]
    assert all(not topics[i].intersection(topics[j]) for i in range(3) for j in range(i))
    train_words = Counter(w for r in rows if r["split"] == "train" for w in r["words"])
    vocab = {w: i + 1 for i, w in enumerate(sorted(train_words, key=lambda w: (-train_words[w], w))[:VOCAB_SIZE - 1])}
    assert len(vocab) + 1 == VOCAB_SIZE
    # All observed types index rows only, without fitting any feature on evaluation text.
    words = sorted({w for r in rows for w in r["words"]})
    row_id = {w: i for i, w in enumerate(words)}
    lookup_ids = np.array([vocab.get(w, 0) for w in words])
    letters = np.zeros((len(words), LENGTH), dtype=np.int32)
    for idx, word in enumerate(words):
        letters[idx, :min(LENGTH, len(word))] = [ord(c) - 96 for c in word[:LENGTH]]
    counts = {}
    for s in SPLITS:
        # Exact sufficient statistics for this one-context-word model. No pair crosses a paragraph.
        pairs = np.zeros((len(words), VOCAB_SIZE))
        for row in rows:
            if row["split"] == s:
                tokens = row["words"]
                for context, target in zip(tokens, tokens[1:]):
                    pairs[row_id[context], vocab.get(target, 0)] += 1
        counts[s] = pairs
    train_observed = np.array([word in train_words for word in words])
    subsets = {"all": np.ones(len(words), dtype=bool), "iv": lookup_ids != 0,
               "oov": lookup_ids == 0, "unseen": ~train_observed}
    split_info = {s: {"topics_with_examples": len(topics[i]),
                      "paragraphs": sum(r["split"] == s for r in rows),
                      "transitions": int(counts[s].sum()),
                      "target_unk": int(counts[s][:, 0].sum())}
                  for i, s in enumerate(SPLITS)}
    info = {"corpus_sha256": CORPUS_HASH, "splits": split_info,
            "test_subset_transitions": {k: int(counts["test"][mask].sum()) for k, mask in subsets.items()},
            "word_types_by_split_union": len(words), "train_word_types": len(train_words),
            "numpy_version": np.__version__}
    return lookup_ids, letters, counts, subsets, info


def initialize(kind, seed):
    # Separate generators make the head exactly identical for the two paired fits.
    common = np.random.default_rng(seed)
    encoder = np.random.default_rng(seed + 10000)
    params = {"head": common.normal(0, 1 / np.sqrt(LENGTH), (LENGTH, VOCAB_SIZE)),
              "bias": np.zeros(VOCAB_SIZE)}
    if kind == "id":
        params["input"] = encoder.normal(0, 0.1, (VOCAB_SIZE, LENGTH))
    elif kind == "char":
        # Row zero is fixed at zero and is not a free parameter.
        params["alphabet"] = encoder.normal(0, 0.3, (26, CHANNELS))
        params["kernel"] = encoder.normal(0, 1 / np.sqrt(3 * CHANNELS), (3 * CHANNELS, CHANNELS))
        params["offset"] = np.zeros(CHANNELS)
        params["reduce"] = encoder.normal(0, 1 / np.sqrt(CHANNELS), CHANNELS)
        params["scalar"] = np.zeros(1)
    return params


def char_encode(letters, p):
    alphabet = np.vstack((np.zeros((1, CHANNELS)), p["alphabet"]))
    embedded = np.pad(alphabet[letters], ((0, 0), (1, 1), (0, 0)))
    windows = np.concatenate([embedded[:, j:j + LENGTH] for j in range(3)], axis=2)
    h = np.tanh(windows @ p["kernel"] + p["offset"])
    z = h @ p["reduce"] + p["scalar"]
    return z, (windows, h)


def char_backward(letters, p, cache, dz):
    windows, h = cache
    dh = dz[:, :, None] * p["reduce"] * (1 - h * h)
    flat_h = dh.reshape(-1, CHANNELS)
    d_windows = (flat_h @ p["kernel"].T).reshape(letters.shape[0], LENGTH, 3, CHANNELS)
    d_embedded = np.zeros((letters.shape[0], LENGTH + 2, CHANNELS))
    for j in range(3):
        d_embedded[:, j:j + LENGTH] += d_windows[:, :, j]
    d_alphabet = np.zeros((27, CHANNELS))
    np.add.at(d_alphabet, letters, d_embedded[:, 1:-1, :])
    return {"alphabet": d_alphabet[1:], "kernel": windows.reshape(-1, 3 * CHANNELS).T @ flat_h,
            "offset": flat_h.sum(axis=0), "reduce": (h * dz[:, :, None]).sum(axis=(0, 1)),
            "scalar": np.array([dz.sum()])}


def loss_grad(kind, p, ids, letters, counts):
    if kind == "id":
        z = p["input"][ids]
    elif kind == "tied":
        z = p["head"].T[ids]
    else:
        z, cache = char_encode(letters, p)
    logits = z @ p["head"] + p["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    dlogits = (probs * counts.sum(axis=1, keepdims=True) - counts) / counts.sum()
    loss = -(counts * np.log(probs)).sum() / counts.sum()
    grads = {"head": z.T @ dlogits, "bias": dlogits.sum(axis=0)}
    dz = dlogits @ p["head"].T
    if kind == "id":
        grads["input"] = np.zeros_like(p["input"])
        np.add.at(grads["input"], ids, dz)
    elif kind == "tied":
        # The shared head also receives gradient through its lookup role.
        np.add.at(grads["head"].T, ids, dz)
    else:
        grads.update(char_backward(letters, p, cache, dz))
    return loss, grads


def nll_by_context(kind, p, ids, letters, counts):
    if kind == "id":
        z = p["input"][ids]
    elif kind == "tied":
        z = p["head"].T[ids]
    else:
        z = char_encode(letters, p)[0]
    logits = z @ p["head"] + p["bias"]
    logits -= logits.max(axis=1, keepdims=True)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    return -(counts * logp).sum(axis=1)


def adam_step(params, grads, m, v, step):
    for name, grad in grads.items():
        m[name] = 0.9 * m[name] + 0.1 * grad
        v[name] = 0.999 * v[name] + 0.001 * grad * grad
        params[name] -= RATE * (m[name] / (1 - 0.9 ** step)) / (np.sqrt(v[name] / (1 - 0.999 ** step)) + 1e-8)


def train(kind, seed, ids, letters, counts, subsets):
    p = initialize(kind, seed)
    n_params = sum(v.size for v in p.values())
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(v) for k, v in p.items()}
    train_mask = counts["train"].sum(axis=1) > 0
    val_mask = counts["val"].sum(axis=1) > 0
    train_ids, train_letters, train_counts = ids[train_mask], letters[train_mask], counts["train"][train_mask]
    val_ids, val_letters, val_counts = ids[val_mask], letters[val_mask], counts["val"][val_mask]
    best_loss = float("inf")
    best_step = None
    best_params = None
    for step in range(1, UPDATES + 1):
        _, grads = loss_grad(kind, p, train_ids, train_letters, train_counts)
        adam_step(p, grads, m, v, step)
        val_loss = nll_by_context(kind, p, val_ids, val_letters, val_counts).sum() / val_counts.sum()
        if val_loss < best_loss:
            best_loss, best_step = val_loss, step
            best_params = {k: a.copy() for k, a in p.items()}
    test_counts = counts["test"]
    nll = nll_by_context(kind, best_params, ids, letters, test_counts)
    result = {"model": kind, "seed": seed, "params": n_params, "step": best_step, "val_ce": best_loss}
    for name, mask in subsets.items():
        result["test_" + name] = nll[mask].sum() / test_counts[mask].sum()
    return result


def save_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    ids, letters, counts, subsets, info = load_data()
    # Add-one smoothed unigram baseline has the same 512-class output space.
    freq = counts["train"].sum(axis=0) + 1
    unigram = freq / freq.sum()
    info["unigram_test_ce"] = float(-(counts["test"] * np.log(unigram)).sum() / counts["test"].sum())
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in SEEDS:
        for kind in ("id", "char", "tied"):
            row = train(kind, seed, ids, letters, counts, subsets)
            rows.append(row)
            print(row, flush=True)
    summary = []
    for name in ("val_ce", "test_all", "test_iv", "test_oov", "test_unseen"):
        base = [r[name] for r in rows if r["model"] == "id"]
        char = [r[name] for r in rows if r["model"] == "char"]
        tied = [r[name] for r in rows if r["model"] == "tied"]
        for key, values in (("id", base), ("char", char), ("tied", tied),
                            ("char_minus_id", [b - a for a, b in zip(base, char)]),
                            ("char_minus_tied", [b - a for a, b in zip(tied, char)])):
            summary.append({"metric": name, "model": key, "mean": statistics.mean(values),
                            "seed_sd": statistics.stdev(values), "n_seeds": len(values)})
    (args.out / "dataset.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    save_csv(args.out / "seeds.csv", rows)
    save_csv(args.out / "summary.csv", summary)


if __name__ == "__main__":
    main()
