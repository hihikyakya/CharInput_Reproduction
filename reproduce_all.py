"""Rebuild and verify the supplemental CPU experiments using the bundled corpus."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent


def run(folder, *args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT / folder, check=True)


def check_values(a, b):
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        assert a == b, (a, b)
    else:
        np.testing.assert_allclose(x, y, rtol=1e-6, atol=1e-12)


def compare_csv(reference, output):
    with reference.open(encoding='utf-8') as a, output.open(encoding='utf-8') as b:
        expected, observed = list(csv.DictReader(a)), list(csv.DictReader(b))
    assert len(expected) == len(observed) and expected
    for row_a, row_b in zip(expected, observed):
        assert row_a.keys() == row_b.keys()
        for key in row_a:
            check_values(row_a[key], row_b[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('rerun'))
    out = parser.parse_args().out.resolve()
    if out.exists():
        parser.error(f'Choose a new output path. Existing directory: {out}')
    out.mkdir(parents=True)
    probe, mse = out / 'ce_probe', out / 'mse_diagnostic'
    run('ce_probe', 'run_probe.py', '--out', probe)
    run('ce_probe', 'check_probe.py', '--compare', probe)
    run('mse_diagnostic', 'reproduce.py', '--out', mse)
    run('mse_diagnostic', 'architecture.py', '--out', mse)
    for name in ['runs.csv', 'summary.csv', 'architecture_components.csv']:
        compare_csv(ROOT / 'mse_diagnostic/results' / name, mse / name)
    for name in ['architecture.json', 'aggregate_audit.json']:
        a = json.loads((ROOT / 'mse_diagnostic/results' / name).read_text())
        b = json.loads((mse / name).read_text())
        assert a == b, name
    run('reference_ce', 'check_protocol.py')
    report = {'ce_probe_fits_checked': 15, 'mse_seeds_checked': 5,
              'architecture_and_aggregate_checks': 'passed',
              'reference_ce_protocol_check': 'passed',
              'original_transformer_training_rerun': False}
    (out / 'verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Verified. Results:', out)


if __name__ == '__main__':
    main()
