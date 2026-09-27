"""Check that both output archives are readable and match the requested sample."""
import json
import zipfile

import numpy as np


def sample_complete(out_dir, prefix, index, params):
    for suffix, required in (
        ("cmb", ("ell", "tt", "ee", "te", "pp", "units")),
        ("derived", ("names", "values", "derived_json")),
    ):
        path = out_dir / f"{prefix}_{index:05d}_{suffix}.npz"
        try:
            with np.load(path, allow_pickle=False) as archive:
                # Read each array to detect truncated archives / bad CRCs.
                arrays = {key: archive[key] for key in required}
                if any(value.size == 0 for value in arrays.values()):
                    return False
                meta = json.loads(archive["meta_json"].item())
                if (meta["run_index"] != index or meta["run"] != prefix
                        or meta["params"] != params):
                    return False
        except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile):
            return False
    return True
