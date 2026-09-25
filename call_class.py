from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_DIR = PROJECT_DIR / "class_public"

PARAMS = {
    "output": "tCl,pCl,lCl",
    "lensing": "yes",

    "h": 0.6737,
    "omega_b": 0.02233,
    "omega_cdm": 0.1198,
    "tau_reio": 0.054,
    "n_s": 0.9652,
    "ln10^{10}A_s": 3.043,
    "N_ncdm": 1,
    "m_ncdm": 0.06,
    "N_ur": 2.0328,

    "sBBN file": str(CLASS_DIR / "bbn/sBBN_2017.dat"),
    "interacting_Cl_file_syn": str(CLASS_DIR / "neutrinos_collision_terms/Coll_integrals_5_qbins.dat"),
    "interacting_Cl_file_new": str(CLASS_DIR / "neutrinos_collision_terms/Coll_integrals_11_qbins.dat"),
    "interacting_alphal_file": str(CLASS_DIR / "neutrinos_collision_terms/Massless_alpha_l.dat"),
    "gauge": "synchronous",
    "log10_G_eff_nu": -12.0,

    "l_max_scalars": 11000,
    "accurate_lensing": 1,
    "k_max_tau0_over_l_max": 15.0,
    "perturb_sampling_stepsize": 0.05,
}

DERIVED_NAMES = [
    "100*theta_s",
    "YHe",
    "z_reio",
    "Neff",
    "tau_rec",
    "z_rec",
    "rs_rec",
    "ra_rec",
    "rs_d",
]

DATA_PARAMETER_MAP = {
    "h": "h",
    "omega_b": "omega_b",
    "omega_cdm": "omega_cdm",
    "ln_A_s_1e10": "ln10^{10}A_s",
    "n_s": "n_s",
    "tau_reio": "tau_reio",
    "N_ur": "N_ur",
    "m_ncdm": "m_ncdm",
    "log10_G_eff_nu": "log10_G_eff_nu",
}


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "output")
    ap.add_argument("--prefix", default="classpt_sinu")
    ap.add_argument("--start", type=int, default=0, help="First run index")
    ap.add_argument("--end", type=int, help="Exclusive final run index (default: all samples)")
    ap.add_argument(
        "--params-file",
        type=Path,
        default=PROJECT_DIR / "dataset/neff_train_param.npz",
        help="NPZ archive containing sampled cosmological parameters",
    )
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.params_file.is_file():
        ap.error(f"parameter file not found: {args.params_file}")

    with np.load(args.params_file, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    missing_keys = set(DATA_PARAMETER_MAP).difference(data)
    if missing_keys:
        ap.error("parameter file is missing: " + ", ".join(sorted(missing_keys)))

    for key in DATA_PARAMETER_MAP:
        if data[key].ndim != 1 or not np.isfinite(data[key]).all():
            ap.error(f"{key} must be a finite one-dimensional array")
    sample_count = len(data["h"])
    if any(len(data[key]) != sample_count for key in DATA_PARAMETER_MAP):
        ap.error("all parameter arrays must have the same length")

    end = sample_count if args.end is None else args.end
    if args.start < 0 or end <= args.start or end > sample_count:
        ap.error(f"choose 0 <= --start < --end <= {sample_count}")

    # Prefer the extension built in this checkout over installed eggs.
    sys.path.insert(0, str(CLASS_DIR / "python"))
    import classy
    from classy import Class

    version = getattr(classy, "__version__", "unknown (CLASS-PT SInu)")
    print(f"Using classy: {classy.__file__}")

    for run_index in range(args.start, end):
        params = dict(PARAMS)
        params.update({
            class_key: data[data_key][run_index].item()
            for data_key, class_key in DATA_PARAMETER_MAP.items()
        })
        print(f"Computing {run_index + 1}/{end}")
        t0 = time.perf_counter()
        cosmo = Class()
        try:
            cosmo.set(params)
            cosmo.compute()

            cls = cosmo.lensed_cl(params["l_max_scalars"])
            ell = np.asarray(cls["ell"][2:], dtype=int)
            tt = cls["tt"][2:]
            ee = cls["ee"][2:]
            te = cls["te"][2:]
            pp = cls["pp"][2:] 
        
            derived = cosmo.get_current_derived_parameters(DERIVED_NAMES)

        finally:
            cosmo.struct_cleanup()
            cosmo.empty()
        runtime = time.perf_counter() - t0

        meta = {
            "code": "class",
            "version": version,
            "classy_module": str(classy.__file__),
            "run": args.prefix,
            "run_index": run_index,
            "runtime_s": runtime,
            "params": params,
        }
        meta_json = np.array(json.dumps(meta))
        run_tag = f"{args.prefix}_{run_index:05d}"
        cmb_path = out_dir / f"{run_tag}_cmb.npz"
        der_path = out_dir / f"{run_tag}_derived.npz"

        np.savez_compressed(
            cmb_path,
            ell=ell,
            tt=tt,
            ee=ee,
            te=te,
            pp=pp,
            units=np.array(json.dumps(
                {"tt": "Cl_dimensionless", "ee": "Cl_dimensionless", "te": "Cl_dimensionless", "pp": "Cl_phi_phi_dimensionless"}
            )),
            meta_json=meta_json,
        )
        np.savez_compressed(
            der_path,
            names=np.array(DERIVED_NAMES),
            values=np.array([derived[name] for name in DERIVED_NAMES], dtype=float),
            derived_json=np.array(json.dumps(derived)),
            meta_json=meta_json,
        )

        print(f"version={version}  runtime={runtime:.1f}s")
        print(f"  {cmb_path}   ell 2..{ell[-1]}")
        print(f"  {der_path}")


if __name__ == "__main__":
    main()
