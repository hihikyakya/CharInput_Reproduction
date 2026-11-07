"""Check computed gradients and compare an independent rerun to the archived report."""
import argparse
import csv
from pathlib import Path

from run_probe import ROOT, np, initialize, loss_grad


def check_gradients():
    counts = np.zeros((3, 512))
    counts[0, [0, 8]] = [1, 2]
    counts[1, 3] = 2
    counts[2, 5] = 1
    ids = np.array([1, 4, 1])
    letters = np.zeros((3, 16), dtype=np.int32)
    letters[:, :3] = np.array([[2, 1, 20], [3, 1, 20], [2, 5, 20]])
    state = np.random.default_rng(97)
    for name in ("id", "char", "tied"):
        p = initialize(name, 11)
        _, grads = loss_grad(name, p, ids, letters, counts)
        eps = 1e-5
        for key, param in p.items():
            direction = state.normal(size=param.shape)
            direction /= np.linalg.norm(direction)
            original = param.copy()
            param[:] = original + eps * direction
            plus, _ = loss_grad(name, p, ids, letters, counts)
            param[:] = original - eps * direction
            minus, _ = loss_grad(name, p, ids, letters, counts)
            param[:] = original
            true_grad = (plus - minus) / (2 * eps)
            np.testing.assert_allclose(true_grad, (grads[key] * direction).sum(), atol=1e-8, rtol=1e-5)
    print("Directional gradient checks passed for every parameter group in all three probes.")


def compare_stats(folder):
    ref = list(csv.DictReader((ROOT / "results/seeds.csv").open(encoding="utf-8")))
    new = list(csv.DictReader((folder / "seeds.csv").open(encoding="utf-8")))
    assert len(ref) == len(new) == 15
    for a, b in zip(ref, new):
        for name in a:
            if name in ("model", "seed", "step", "params"):
                assert a[name] == b[name], (a, b, name)
            else:
                np.testing.assert_allclose(float(a[name]), float(b[name]), atol=1e-7, rtol=1e-7)
    print("All 15 fit records match the archived results within 1e-7.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    check_gradients()
    if args.compare:
        compare_stats(args.compare)
