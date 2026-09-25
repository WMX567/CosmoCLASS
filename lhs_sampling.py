"""Generate reproducible, independent Latin hypercube designs for each split."""
import argparse
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
PARAM_RANGES = {
    "omega_b": (0.015, 0.03),
    "omega_cdm": (0.09, 0.15),
    "ln_A_s_1e10": (2.5, 3.5),
    "n_s": (0.85, 1.05),
    "h": (0.4, 1.0),
    # N_ur is the massless contribution, not the total N_eff.
    "N_ur": (1.0, 4.0),
    # One massive species, default degeneracy 1; mass in eV.
    "m_ncdm": (1e-4, 1.5),
    "log10_G_eff_nu": (-7.0, -0.5),
    "tau_reio": (0.02, 0.20),
}


def perform_lhs_sampling(param_ranges, n_samples, seed=None):
    """Each column visits every equal-width stratum exactly once."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    rng = np.random.default_rng(seed)
    sampled = {}
    for name, (lower, upper) in param_ranges.items():
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(f"Invalid range for {name}")
        unit = (rng.permutation(n_samples) + rng.random(n_samples)) / n_samples
        sampled[name] = lower + (upper - lower) * unit
    return sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=60000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--val-size", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "dataset")
    args = parser.parse_args()
    sizes = (args.train_size, args.test_size, args.val_size)
    if min(sizes) <= 0 or args.seed < 0:
        parser.error("split sizes must be positive and seed nonnegative")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = np.random.SeedSequence(args.seed).spawn(3)
    for split, size, seed in zip(("train", "test", "val"), sizes, seeds):
        samples = perform_lhs_sampling(PARAM_RANGES, size, seed)
        path = args.out_dir / f"neff_{split}_param.npz"
        np.savez_compressed(path, **samples)
        print(f"{split}: {size} samples -> {path}")


if __name__ == "__main__":
    main()
