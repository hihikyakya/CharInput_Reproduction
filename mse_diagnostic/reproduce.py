#!/usr/bin/env python3
"""Fixed-target negative control and a separate audit of published aggregates."""
from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np


SEEDS = (17, 29, 43, 59, 71)
VOCAB = 16
LENGTH = 17
N_TRAIN, N_TEST = 1024, 256
TARGET_SCALE = 0.25
HERE = Path(__file__).resolve().parent


def token_windows(rng: np.random.Generator, n_sequences: int):
    # Uniform i.i.d. tokens; pairs do not cross sequence boundaries.
    tokens = rng.integers(0, VOCAB, size=(n_sequences, LENGTH))
    return tokens[:, :-1].ravel(), tokens[:, 1:].ravel()


EYE = np.eye(VOCAB)


def features(now, following, allow_future):
    # Drop last contrast to obtain an identifiable model with an intercept.
    current = EYE[now, :-1]
    design = [np.ones((len(now), 1)), current]
    if allow_future:
        design.append(EYE[following, :-1])
    return np.column_stack(design)


def loss(predicted, target):
    return float(np.mean((predicted - target) ** 2))


def run_trial(seed: int):
    rng = np.random.Generator(np.random.PCG64(seed))
    train_now, train_next = token_windows(rng, N_TRAIN)
    test_now, test_next = token_windows(rng, N_TEST)
    y_train, y_test = EYE[train_next], EYE[test_next]
    train_mean = y_train.mean(axis=0)
    mean_loss = loss(train_mean, y_test)
    rows = []
    for label, allow_future in (("causal", False), ("future_visible", True)):
        x_train = features(train_now, train_next, allow_future)
        x_test = features(test_now, test_next, allow_future)
        weights = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        pred = x_test @ weights

        # Keep the current token fixed, then replace the next token in every pair.
        changed = features(test_now, (test_next + 1) % VOCAB, allow_future)
        disturbance = float(np.max(np.abs(changed @ weights - pred)))
        mse = loss(pred, y_test)
        nmse = mse / mean_loss
        # Refit and score in a rescaled target space, preserving all splits/features.
        scaled_weights = np.linalg.lstsq(x_train, TARGET_SCALE * y_train, rcond=None)[0]
        scaled_mse = loss(x_test @ scaled_weights, TARGET_SCALE * y_test)
        scaled_nmse = scaled_mse / loss(TARGET_SCALE * train_mean, TARGET_SCALE * y_test)
        np.testing.assert_allclose(scaled_mse, TARGET_SCALE**2 * mse, atol=1e-14)
        np.testing.assert_allclose(scaled_nmse, nmse, atol=1e-13)

        if allow_future:
            assert mse < 1e-24 and abs(disturbance - 1) < 1e-12
        else:
            assert disturbance == 0
        rows.append({
            "seed": seed, "available_context": label, "mse": mse,
            "train_mean_mse": mean_loss, "nmse": nmse, "scaled_mse": scaled_mse,
            "scaled_nmse": scaled_nmse, "suffix_max_change": disturbance,
        })
    return rows


def summary(rows):
    summary_rows = []
    for mode in ("causal", "future_visible"):
        selected = [r for r in rows if r["available_context"] == mode]
        result = {"available_context": mode, "n_seeds": len(selected)}
        for field in ("mse", "train_mean_mse", "nmse", "scaled_mse", "scaled_nmse", "suffix_max_change"):
            values = [r[field] for r in selected]
            result[field + "_mean"] = float(np.mean(values))
            result[field + "_sample_sd"] = float(np.std(values, ddof=1))
        summary_rows.append(result)
    return summary_rows


def aggregate_audit():
    source = json.loads((HERE / "reported_inputs.json").read_text(encoding="utf-8"))
    dim = source["target_dimension"]
    b, c = source["token_id"], source["char_encoder"]
    norm_b = dim * b["lambada_mse"] / b["target_total_variance"]
    norm_c = dim * c["lambada_mse"] / c["target_total_variance"]
    var_ratio = b["target_total_variance"] / c["target_total_variance"]
    raw_ratio = b["lambada_mse"] / c["lambada_mse"]
    assert abs(raw_ratio - var_ratio * norm_b / norm_c) < 1e-12
    for reconstructed, orig in ((norm_b, b), (norm_c, c)):
        assert abs(reconstructed - orig["test_mean_nmse_rounded"]) < 0.01
    return {
        "raw_mse_baseline_over_char": raw_ratio,
        "total_variance_baseline_over_char": var_ratio,
        "implied_test_mean_nmse_token": norm_b,
        "implied_test_mean_nmse_char": norm_c,
        "implied_nmse_baseline_over_char": norm_b / norm_c,
        "validation_evalmask_mse_increase_token": b["validation_masked_mse"] / b["validation_unmasked_mse"],
        "validation_evalmask_mse_increase_char": c["validation_masked_mse"] / c["validation_unmasked_mse"],
    }


def csv_dump(rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=rows[0], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [row for seed in SEEDS for row in run_trial(seed)]
    summary_rows = summary(rows)
    audit = aggregate_audit()
    for name, data in (("runs", rows), ("summary", summary_rows)):
        (args.out / (name + ".csv")).write_text(csv_dump(data), encoding="utf-8")
    (args.out / "aggregate_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    for r in summary_rows:
        print(f"{r['available_context']:>14}: NMSE {r['nmse_mean']:.6f} +/- "
              f"{r['nmse_sample_sd']:.6f}, suffix max {r['suffix_max_change_mean']:.6f}")
    print(f"Reported MSE ratio {audit['raw_mse_baseline_over_char']:.3f} = "
          f"{audit['total_variance_baseline_over_char']:.3f} x "
          f"{audit['implied_nmse_baseline_over_char']:.3f}")


if __name__ == "__main__":
    main()
