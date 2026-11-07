"""Regenerate the local English next-word probe from CPython's topic documentation."""
from pathlib import Path
import gzip
import hashlib
import json
import re
from pydoc_data.topics import topics

ROOT = Path(__file__).resolve().parent
SPLIT_SALT = "word-input-study-v1"
DEDUP = set()


def paragraphs(topic):
    for chunk in re.split(r"\n\s*\n", topic):
        if not chunk or not chunk[0].isalpha() or "\n***" in chunk or "\n===" in chunk:
            continue
        words = tuple(re.findall(r"[a-z]+", chunk.lower()))
        if len(words) >= 8 and words not in DEDUP:
            DEDUP.add(words)
            yield words


def main():
    # Choose a deterministic topic split before looking at text or outcomes.
    names = sorted(topics, key=lambda s: hashlib.sha256((SPLIT_SALT + "/" + s).encode()).digest())
    assignments = {}
    for i, key in enumerate(names):
        assignments[key] = "train" if i < 59 else ("val" if i < 69 else "test")
    rows = []
    for key in sorted(topics):
        for words in paragraphs(topics[key]):
            rows.append({"topic": key, "split": assignments[key], "words": words})
    raw = "".join(json.dumps(r, ensure_ascii=True, separators=(",", ":")) + "\n" for r in rows).encode()
    (ROOT / "data/probe.jsonl.gz").write_bytes(gzip.compress(raw, compresslevel=9, mtime=0))
    print("corpus SHA256 (uncompressed):", hashlib.sha256(raw).hexdigest())


if __name__ == "__main__":
    main()
