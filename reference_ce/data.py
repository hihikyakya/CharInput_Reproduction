"""Shared, dependency-free protocol helpers for the optional CE rerun."""
import json
import random
import string
from collections import Counter
from itertools import islice

PAD, UNK, EOS = '<PAD>', '<UNK>', '<EOS>'
SPECIAL = (PAD, UNK, EOS, '<BOS>', '<MASK>', '<SEP>')
VOCAB_SIZE, WIDTH, CONTEXT = 20_000, 128, 64
CHAR = {c: i + 1 for i, c in enumerate(string.ascii_lowercase + string.digits + string.punctuation + ' ')}


def char_row(token):
    letters = [CHAR.get(ch, 0) for ch in token.lower()[:WIDTH]]
    return letters + [0] * (WIDTH - len(letters))


def partition(path, seed):
    # Zero-based indices refer to the selected nonempty lines, not physical line numbers.
    with open(path, encoding='utf-8') as source:
        lines = list(islice((s.strip() for s in source if s.strip()), 5000))
    if len(lines) != 5000:
        raise ValueError('The WikiText training file must contain at least 5,000 nonempty lines.')
    indices = list(range(5000))
    random.Random(seed).shuffle(indices)
    return lines, indices[:3500], indices[3500:]


def make_vocab(token_lists):
    counts = Counter(t for seq in token_lists for t in seq if t not in SPECIAL)
    ordered = sorted(counts, key=lambda t: (-counts[t], t))[:VOCAB_SIZE - len(SPECIAL)]
    return {t: i for i, t in enumerate((*SPECIAL, *ordered))}


def shifted(words):
    seq = [*words, EOS][:CONTEXT + 1]
    return seq[:-1], seq[1:]


def passages(path):
    """Plain text, one passage per line; JSONL with a text key is also accepted."""
    with open(path, encoding='utf-8') as inp:
        for lineno, line in enumerate(inp, 1):
            if not line.strip():
                continue
            if str(path).lower().endswith(('.json', '.jsonl')):
                line = json.loads(line)['text']
            pieces = line.strip().rsplit(None, 1)
            if len(pieces) != 2:
                raise ValueError(f'LAMBADA line {lineno}: expected context and final whitespace-delimited word.')
            yield pieces
