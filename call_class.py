from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent
CLASS_DIR = PROJECT_DIR / "class_public"

PARAMS = {
    "output": "tCl,pCl,lCl,mPk",
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
    "non linear": "halofit",

    "l_max_scalars": 11000,
    "P_k_max_1/Mpc": 50.0,
    "accurate_lensing": 1,
    "k_max_tau0_over_l_max": 15.0,
    "perturb_sampling_stepsize": 0.05,
    "halofit_min_k_max": 100,
}

DERIVED_NAMES = [
    "100*theta_s",
    "sigma8",
    "YHe",
    "z_reio",
    "Neff",
    "tau_rec",
    "z_rec",
    "rs_rec",
    "ra_rec",
    "rs_d",
]

K_MIN = 1e-4
K_MAX_OUT = 50.0
N_K = 500

# Redshifts at which P(k) is evaluated and stored for every sample.
PK_REDSHIFTS_DEFAULT = [0.0, 2.5, 5.0]

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


def z_tag(z: float) -> str:
    return "z" + f"{z:g}".replace(".", "p")


def main() -> None:

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pk-redshifts",
        type=float,
        nargs="+",
        default=PK_REDSHIFTS_DEFAULT,
        help="Redshifts at which P_k_lin and P_k_nl are stored (default: %(default)s)",
    )
    ap.add_argument(
        "--z-max-pk",
        type=float,
        default=None,
        help=(
            "Upper redshift limit CLASS uses to build the P(k) table "
            "(class 'z_max_pk'). Must be >= max(--pk-redshifts). "
            "Defaults to max(--pk-redshifts)."
        ),
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR / "output")
    ap.add_argument("--prefix", default="classpt_sinu_halofit")
    ap.add_argument("--start", type=int, default=0, help="First run index")
    ap.add_argument("--end", type=int, help="Exclusive final run index (default: all samples)")
    ap.add_argument(
        "--params-file",
        type=Path,
        default=PROJECT_DIR / "dataset/neff_train_param.npz",
        help="NPZ archive containing sampled cosmological parameters",
    )
    args = ap.parse_args()

    if any(not np.isfinite(z) or z < 0 for z in args.pk_redshifts):
        ap.error("--pk-redshifts must be finite and nonnegative")
    pk_redshifts = sorted(set(args.pk_redshifts))
    z_max_pk = args.z_max_pk if args.z_max_pk is not None else max(pk_redshifts)
    if not np.isfinite(z_max_pk) or z_max_pk < max(pk_redshifts):
        ap.error(
            f"--z-max-pk ({z_max_pk}) must be >= max(--pk-redshifts) ({max(pk_redshifts)})"
        )

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

    import classy
    from classy import Class

    if not hasattr(Class, "pk_halofit"):
        ap.error("Rebuild and install class_public/python: this driver requires the local pk_halofit interface")

    version = getattr(classy, "__version__", "unknown (CLASS-PT SInu)")
    print(f"Using classy: {classy.__file__}")

    k = np.logspace(np.log10(K_MIN), np.log10(K_MAX_OUT), N_K)

    for run_index in range(args.start, end):
        params = dict(PARAMS)
        params.update({
            class_key: data[data_key][run_index].item()
            for data_key, class_key in DATA_PARAMETER_MAP.items()
        })
        # z_max_pk only sets how far CLASS builds its internal P(k) table;
        # the actual redshifts we sample from that table are pk_redshifts.
        params["z_max_pk"] = z_max_pk

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
        
            # One P(k) curve per requested redshift, all pulled from the same
            # z_max_pk table computed above.
            pk_lin_by_z = {z: np.array([cosmo.pk_lin(ki, z) for ki in k]) for z in pk_redshifts}
            pk_nl_by_z = {z: np.array([cosmo.pk_halofit(ki, z) for ki in k]) for z in pk_redshifts}
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
            "pk_redshifts": pk_redshifts,
            "z_max_pk": z_max_pk,
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

        pk_paths = []
        for z in pk_redshifts:
            pk_path = out_dir / f"{run_tag}_{z_tag(z)}_pk.npz"
            np.savez_compressed(
                pk_path,
                k=k,
                z=np.array(z),
                pk_lin=pk_lin_by_z[z],
                pk_nl=pk_nl_by_z[z],
                units=np.array(json.dumps(
                    {"k": "1/Mpc", "pk_lin": "Mpc3", "pk_nl": "Mpc3"}
                )),
                meta_json=meta_json,
            )
            pk_paths.append(pk_path)

        print(f"version={version}  z_max_pk={z_max_pk}  pk_redshifts={pk_redshifts}  runtime={runtime:.1f}s")
        print(f"  {cmb_path}   ell 2..{ell[-1]}")
        for pk_path, z in zip(pk_paths, pk_redshifts):
            print(f"  {pk_path}    k {k[0]:.1e}..{k[-1]:g} 1/Mpc, z={z}")
        print(f"  {der_path}")


if __name__ == "__main__":
    main()
