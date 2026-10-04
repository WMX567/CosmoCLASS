"""Compute matter power spectra P(k) for LHS samples not covered by call_class.py.

call_class.py runs the self-interacting-neutrino CLASS-PT branch with
output=tCl,pCl,lCl only; it never requests mPk, so no P(k) exists for any of
the 68000 LHS samples. This driver reuses the same sample loop, parameter
file, and self-interacting-neutrino physics settings, but requests mPk with
HMcode nonlinear corrections and saves linear/nonlinear P(k) per sample.

Usage:
    python call_class_pk.py --pk-z 0 0.5 1 2 3 \
        --start 0 --end 2000 --params-file dataset/neff_train_param.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from call_class import CLASS_DIR, DATA_PARAMETER_MAP, PROJECT_DIR, peak_rss_mib

PARAMS = {
    "output": "mPk",

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

    "non linear": "hmcode",
    "c_min": 3.13,
    "eta_0": 0.603,
    "P_k_max_1/Mpc": 50.0,
    "perturb_sampling_stepsize": 0.05,
    "nonlinear_min_k_max": 100,
    "hmcode_min_k_max": 100,
}

# sigma8 is defined at z=0 and only becomes available once mPk is requested;
# the thermodynamics-derived parameters (theta_s, z_rec, rs_d, ...) are
# already saved by call_class.py and are not duplicated here.
DERIVED_NAMES = ["sigma8"]

K_MIN = 1e-4
K_MAX_OUT = 50.0
N_K = 500


def z_tag(z: float) -> str:
    return "z" + f"{z:g}".replace(".", "p")


def pk_sample_complete(path: Path, index: int, prefix: str, params: dict, pk_z: float) -> bool:
    """Mirror sample_status.sample_complete, for a single-z P(k) archive."""
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in ("k", "pk_lin", "pk_nl", "z")}
            if any(np.asarray(value).size == 0 for value in arrays.values()):
                return False
            if float(arrays["z"]) != pk_z:
                return False
            meta = json.loads(archive["meta_json"].item())
            if (meta["run_index"] != index or meta["run"] != prefix
                    or meta["params"] != params):
                return False
    except (OSError, ValueError, KeyError, EOFError, TypeError, zipfile.BadZipFile):
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "output")
    ap.add_argument("--prefix", default="classpt_sinu")
    ap.add_argument("--pk-z", type=float, nargs="+", default=[0.0],
                     help="Redshift(s) at which to evaluate P(k); one output file per z")
    ap.add_argument("--resume", action="store_true",
                     help="Skip readable P(k) outputs whose saved parameters match")
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
    print(f"Using classy: {classy.__file__}", flush=True)

    k = np.logspace(np.log10(K_MIN), np.log10(K_MAX_OUT), N_K)
    z_max_pk = max(args.pk_z)

    for run_index in range(args.start, end):
        params = dict(PARAMS)
        params.update({
            class_key: data[data_key][run_index].item()
            for data_key, class_key in DATA_PARAMETER_MAP.items()
        })
        if z_max_pk > 0:
            params["z_max_pk"] = z_max_pk

        pk_paths = {
            z: out_dir / f"{args.prefix}_{run_index:05d}_pk_{z_tag(z)}.npz"
            for z in args.pk_z
        }
        if args.resume and all(
            pk_sample_complete(pk_paths[z], run_index, args.prefix, params, z)
            for z in args.pk_z
        ):
            print(f"Skipping completed index={run_index}", flush=True)
            continue

        print(f"Computing index={run_index} ({run_index + 1}/{end}); "
              f"process_peak_rss_mib={peak_rss_mib():.1f}", flush=True)
        t0 = time.perf_counter()
        cosmo = Class()
        try:
            cosmo.set(params)
            cosmo.compute()
            print(f"CLASS compute finished: index={run_index}; "
                  f"process_peak_rss_mib={peak_rss_mib():.1f}", flush=True)

            derived = cosmo.get_current_derived_parameters(DERIVED_NAMES)
            pk_by_z = {}
            for z in args.pk_z:
                pk_lin = np.array([cosmo.pk_lin(ki, z) for ki in k])
                pk_nl = np.array([cosmo.pk(ki, z) for ki in k])
                pk_by_z[z] = (pk_lin, pk_nl)
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
            "process_peak_rss_mib": peak_rss_mib(),
            "params": params,
        }
        meta_json = np.array(json.dumps(meta))

        for z, (pk_lin, pk_nl) in pk_by_z.items():
            pk_path = pk_paths[z]
            np.savez_compressed(
                pk_path,
                k=k,
                pk_lin=pk_lin,
                pk_nl=pk_nl,
                z=np.array(z),
                names=np.array(DERIVED_NAMES),
                values=np.array([derived[name] for name in DERIVED_NAMES], dtype=float),
                derived_json=np.array(json.dumps(derived)),
                units=np.array(json.dumps(
                    {"k": "1/Mpc", "pk_lin": "Mpc3", "pk_nl": "Mpc3"}
                )),
                meta_json=meta_json,
            )
            print(f"  {pk_path}    k {k[0]:.1e}..{k[-1]:g} 1/Mpc  z={z:g}")

        print(f"version={version}  runtime={runtime:.1f}s  "
              f"process_peak_rss_mib={peak_rss_mib():.1f}", flush=True)


if __name__ == "__main__":
    main()
