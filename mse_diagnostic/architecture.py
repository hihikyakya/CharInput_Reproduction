#!/usr/bin/env python3
"""Reconstruct trainable parameters and dense CharEnc MACs from Sec. 2."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D, SLOTS, ALPHABET, VOCAB, LAYERS = 128, 128, 70, 20_000, 8
CHANNELS, BLOCKS_PER_STAGE = (64, 32, 16), 2


def char_encoder():
    rows = []
    def add(name, params, macs_per_position):
        rows.append(dict(component=name, parameters=params,
                         macs_per_full_length_token=SLOTS * macs_per_position))
    add("character lookup", ALPHABET * D, 0)
    prev = D
    for channels in CHANNELS:
        # Width-1 projection, bias, per-channel LayerNorm affine parameters.
        proj = prev * channels
        add(f"{prev} to {channels} projection and LN", proj + 3 * channels, proj)
        # Width-3 convolution and width-1 convolution, one bias/channel each;
        # residual addition + GELU are parameter-free. One affine LayerNorm.
        block_weights = (3 + 1) * channels**2
        add(f"two width-{channels} residual blocks with LN",
            BLOCKS_PER_STAGE * (block_weights + 4 * channels),
            BLOCKS_PER_STAGE * block_weights)
        prev = channels
    add("final scalar projection", prev + 1, prev)
    return rows


def model_audit():
    components = char_encoder()
    char_params = sum(r["parameters"] for r in components)
    char_macs = sum(r["macs_per_full_length_token"] for r in components)
    # Q/K/V plus output (weights, biases), 4D FFN, two affine LayerNorms.
    attn = 4 * (D * D + D)
    feedforward = D * (4 * D) + 4 * D + (4 * D) * D + D
    norms = 2 * (2 * D)
    encoder = LAYERS * (attn + feedforward + norms)
    output_head = VOCAB * (D + 1)
    shared_body_and_head = encoder + output_head
    token_table = VOCAB * D
    token_untied = shared_body_and_head + token_table
    word_untied = shared_body_and_head + char_params
    # A tied token table and the output matrix have one trainable copy.
    token_tied = token_untied - token_table
    # Every residual block contains one kernel-3 convolution; other
    # convolutions have kernel size 1. All convolutions have unit stride/dilation.
    receptive_field = 1 + len(CHANNELS) * BLOCKS_PER_STAGE * (3 - 1)
    result = {
        "input_params_token_id": token_table, "input_params_char": char_params,
        "transformer_params": encoder, "output_params_including_bias": output_head,
        "untied_token_model_total": token_untied, "char_model_total": word_untied,
        "char_total_reduction_percent_vs_untied_token": 100 * (1 - word_untied / token_untied),
        "char_input_reduction_percent_vs_token_input": 100 * (1 - char_params / token_table),
        "hypothetical_tied_token_total": token_tied,
        "char_total_excess_vs_tied_token_params": word_untied - token_tied,
        "char_total_excess_percent_vs_tied_token": 100 * (word_untied - token_tied) / token_tied,
        "char_full_token_convolution_and_projection_macs": char_macs,
        "char_maximum_theoretical_receptive_field_slots": receptive_field,
    }
    assert (char_params, encoder, token_untied, word_untied) == (63_969, 1_586_176, 6_726_176, 4_230_145)
    assert (token_tied, char_macs, receptive_field) == (4_166_176, 6_883_328, 13)
    return components, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    components, result = model_audit()
    with (args.out / "architecture_components.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=components[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(components)
    (args.out / "architecture.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
