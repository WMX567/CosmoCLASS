"""Replace zero-based training row 44140 with the separately generated sample."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from lhs_sampling import PARAM_RANGES

ROOT = Path(__file__).resolve().parent
INDEX = 44140


def update(target, replacement):
    with np.load(target, allow_pickle=False) as archive:
        original = {key: archive[key] for key in archive.files}
    with np.load(replacement, allow_pickle=False) as archive:
        sampled = {key: archive[key] for key in archive.files}
    for key in PARAM_RANGES:
        if key not in original or key not in sampled:
            raise ValueError(f"Missing parameter: {key}")
        if original[key].ndim != 1 or len(original[key]) <= INDEX:
            raise ValueError(f"Training array {key} has no row {INDEX}")
        if original[key].dtype.kind != "f":
            raise ValueError(f"Training array {key} must have a floating dtype")
        if sampled[key].shape != (1,) or not np.isfinite(sampled[key]).all():
            raise ValueError(f"Replacement {key} must contain one finite value")
        lower, upper = PARAM_RANGES[key]
        if not lower <= sampled[key][0] <= upper:
            raise ValueError(f"Replacement {key} is outside its sampling range")
    if len({len(original[key]) for key in PARAM_RANGES}) != 1:
        raise ValueError("Training parameter arrays have different lengths")

    updated = {key: value.copy() for key, value in original.items()}
    for key in PARAM_RANGES:
        updated[key][INDEX] = sampled[key][0]
        if updated[key][INDEX] != sampled[key][0]:
            raise ValueError(f"Replacement {key} loses precision in training dtype")
    if all(np.array_equal(original[key], updated[key], equal_nan=True)
           for key in PARAM_RANGES):
        print(f"Row {INDEX} already matches; no changes made.")
        return

    backup = target.with_name(target.name + ".before_44140.bak")
    # Exclusive creation protects the original backup on repeated invocations.
    with backup.open("xb") as dest, target.open("rb") as source:
        shutil.copyfileobj(source, dest)
    print(f"Backup: {backup}")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".npz", delete=False) as handle:
            temp_path = Path(handle.name)
            np.savez_compressed(handle, **updated)
        with np.load(temp_path, allow_pickle=False) as check:
            if set(check.files) != set(original):
                raise ValueError("Archive keys changed")
            for key, before in original.items():
                after = check[key]
                if after.shape != before.shape or after.dtype != before.dtype:
                    raise ValueError(f"Shape or dtype changed: {key}")
                expected = updated[key]
                if not np.array_equal(after, expected):
                    raise ValueError(f"Saved values do not match: {key}")
                if key in PARAM_RANGES:
                    after = after.copy()
                    after[INDEX] = before[INDEX]
                if not np.array_equal(after, before):
                    raise ValueError(f"Other rows changed: {key}")
        os.chmod(temp_path, target.stat().st_mode & 0o777)
        os.replace(temp_path, target)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
    print(f"Updated {target}: zero-based index {INDEX}; all other rows unchanged.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params-file", type=Path, default=ROOT / "dataset/neff_train_param.npz")
    parser.add_argument("--replacement-file", type=Path,
                        default=ROOT / "dataset/replacement_44140/params.npz")
    args = parser.parse_args()
    try:
        update(args.params_file, args.replacement_file)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
