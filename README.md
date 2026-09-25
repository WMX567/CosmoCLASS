# CosmoCLASS

Generate cosmological parameters with Latin hypercube sampling (LHS) and compute CMB power spectra, linear and Halofit nonlinear matter power spectra, and derived parameters using the local CLASS-PT branch for self-interacting neutrinos.

## Repository layout

| Path | Purpose |
| --- | --- |
| `lhs_sampling.py` | Generate training, test, and validation parameter sets |
| `call_class.py` | Run CLASS for a range of sample indices and save results |
| `class_public/` | CLASS-PT source code and Python interface for self-interacting neutrinos |
| `dataset/` | Default parameter directory, created during sampling |
| `output/` | Default results directory, created during computation |
| `data_generator.sh` | Slurm sampling job |
| `data1.sh` … `data30.sh` | Thirty training jobs, 2000 samples per job |
| `test_data1.sh` … `test_data2.sh`, `val_data1.sh` … `val_data2.sh` | Two jobs per split, 2000 samples per job |
| `start.sh` | Submit all thirty training jobs |

The data-generation entry points and job scripts live in the project root. Duplicate Python entry points, the obsolete job generator, and unused legacy logging and plotting scripts have been removed.

## Model and interface

See [class_public/README.md](class_public/README.md) for details about this CLASS branch. The current driver fixes `N_ncdm=1` and uses synchronous gauge.

- `m_ncdm` is the mass of one massive neutrino species in eV, with the default degeneracy of 1. It does not automatically configure three equal-mass species.
- `N_ur` specifies the massless contribution, rather than the total effective number of neutrinos. The actual total is saved as the derived parameter `Neff`.
- `log10_G_eff_nu` controls neutrino self-interactions. Sampled values are passed directly to CLASS.
- This version does not support HMcode. The driver uses `non linear = halofit`; producing these spectra does not establish Halofit accuracy throughout the sampled self-interaction parameter range.

The original `pk()` in this branch returns a list of perturbation-theory (PT) components, rather than the scalar nonlinear power spectrum returned by standard CLASS. This project adds `pk_halofit()` in `class_public/python/classy.pyx`, with its declaration in `cclassy.pxd`, to call the existing C Halofit interpolation function. The driver uses this interface to save `pk_nl`.

## Environment setup

The sampling script requires only Python and NumPy. Spectrum computation also requires `classy` compiled from this project's source. The extension depends on SciPy; building it requires Cython, NumPy, a C compiler, and the appropriate OpenBLAS/OpenMP libraries.

**Rebuild and install the local extension. Standard CLASS and the old binaries included in the repository cannot replace it.** The driver checks for `pk_halofit()` and prints the path of the imported `classy` module.

Before building, adjust these settings for the target machine:

| File | Settings to check |
| --- | --- |
| `class_public/Makefile` | `CC`, `OPENBLAS`, `OMPFLAG`, `PYTHON` |
| `class_public/python/setup.py` | OpenBLAS/OpenMP paths in `library_dirs` and `extra_link_args` |
| Job scripts | Load `conda.sh`, then run `conda activate multinest` |

The current build configuration contains paths under `/home1/mengxiwu/.conda/envs/multinest/` from the original cluster. These paths must be checked on other machines. Compiler, shared-library, and OpenMP configurations also differ between macOS and Linux.

After updating the configuration, build in the intended Python environment. Clean any build artifacts copied from another machine first:

```bash
cd class_public
make clean
make all
cd ..

python -c 'import classy; from classy import Class; print(classy.__file__); assert hasattr(Class, "pk_halofit")'
```

`make all` includes installation of the Python extension. Check the printed path to confirm that Python imports the newly built extension. Some CLASS resource paths are bound to the source directory at compile time, so rebuild after moving the source tree.

## Generate parameter sets

Run from the project root:

```bash
python lhs_sampling.py --seed 42
```

By default, this creates the following files. Each contains nine one-dimensional parameter arrays of equal length:

| File | Samples |
| --- | ---: |
| `dataset/neff_train_param.npz` | 60000 |
| `dataset/neff_test_param.npz` | 4000 |
| `dataset/neff_val_param.npz` | 4000 |

The parameter ranges are defined in `PARAM_RANGES` in `lhs_sampling.py`:

| Parameter | Lower bound | Upper bound |
| --- | ---: | ---: |
| `omega_b` | 0.015 | 0.03 |
| `omega_cdm` | 0.09 | 0.15 |
| `ln_A_s_1e10` | 2.5 | 3.5 |
| `n_s` | 0.85 | 1.05 |
| `h` | 0.4 | 1.0 |
| `N_ur` | 1.0 | 4.0 |
| `m_ncdm` | 0.0001 | 1.5 |
| `log10_G_eff_nu` | -7.0 | -0.5 |
| `tau_reio` | 0.02 | 0.20 |

`ln_A_s_1e10` maps to the CLASS parameter `ln10^{10}A_s`, the natural logarithm ln(10¹⁰ A_s). Mass is sampled in linear space, while the coupling parameter is sampled in log10 space.

Each split receives a separate nine-dimensional LHS design. For a split with N samples, each parameter range is divided into N equal-width strata, with one random draw per stratum. Independent permutations across dimensions produce N parameter rows without constructing a Cartesian product. The same seed and settings reproduce the results; rerunning overwrites parameter files with the same names.

For a small trial, use a separate directory:

```bash
python lhs_sampling.py \
  --train-size 10 --test-size 2 --val-size 2 \
  --seed 42 --out-dir dataset/smoke
```

## Compute spectra

After installing the extension, compute one sample to check the environment:

```bash
python call_class.py \
  --params-file dataset/smoke/neff_train_param.npz \
  --start 0 --end 1 \
  --pk-redshifts 0 2.5 5 \
  --out-dir output/smoke
```

This reduces the number of samples but retains the production precision settings.

| Option | Default and meaning |
| --- | --- |
| `--params-file` | `dataset/neff_train_param.npz` under the project root |
| `--start` | 0; inclusive starting index |
| `--end` | Number of samples in the parameter file; exclusive ending index |
| `--pk-redshifts` | `0 2.5 5`; accepts multiple nonnegative redshifts |
| `--z-max-pk` | Largest requested redshift by default; an explicit value must be at least this large |
| `--out-dir` | `output/` under the project root |
| `--prefix` | `classpt_sinu_halofit` |

Default paths are relative to the project root. Relative paths explicitly passed on the command line are resolved against the current working directory. `--start 0 --end 1` computes only sample 0.

The current configuration uses `l_max_scalars=11000`. Matter spectra are saved at 500 logarithmically spaced k values from 10⁻⁴ to 50 Mpc⁻¹. Precision settings and output grids are defined in `call_class.py`.

## Submit Slurm jobs

Job scripts initialize Conda in the job shell and activate the environment:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate multinest
```

The `conda` command must be available on `PATH`, and `multinest` must contain the local CLASS extension. Each job loads the initialization script explicitly, so it does not depend on the batch shell reading an interactive shell configuration. The scripts use `set -eo pipefail`, without `set -u`, to allow Conda compiler activation hooks to read unset variables.

```bash
export COSMOCLASS_DIR=/path/to/CosmoCLASS
sbatch "$COSMOCLASS_DIR/data_generator.sh"
```

Wait for sampling to finish successfully and confirm that the parameter files exist, then submit the computation jobs:

```bash
bash "$COSMOCLASS_DIR/start.sh"
sbatch "$COSMOCLASS_DIR/test_data1.sh"
sbatch "$COSMOCLASS_DIR/test_data2.sh"
sbatch "$COSMOCLASS_DIR/val_data1.sh"
sbatch "$COSMOCLASS_DIR/val_data2.sh"
```

`start.sh` submits only the training jobs. It does not wait for sampling or submit the test and validation jobs.

Training script `dataN.sh` processes indices `[2000 × (N−1), 2000 × N)`. The thirty jobs cover all 60000 training samples without gaps or overlap. Each of the test and validation splits uses two jobs covering `[0, 2000)` and `[2000, 4000)`, with prefixes `classpt_sinu_halofit_test` and `classpt_sinu_halofit_val`, respectively. There are 34 spectrum-computation jobs in total. The separate sampling job creates all three parameter files. If you change the sample counts, update the job index ranges and the loop in `start.sh` accordingly.

Scripts locate the project using `COSMOCLASS_DIR` first, then `SLURM_SUBMIT_DIR`, or their own directory during ordinary Bash execution. Since Slurm copies job scripts, explicitly set `COSMOCLASS_DIR` when submitting from another directory.

Results are written to the project's `output/` directory. Relative Slurm log paths are resolved against the job working directory set at submission, and are unaffected by `cd` inside the script. The batch training launcher explicitly sets the job working directory to the project root.

`class_public/reproduce.sh` runs the original `david_test.py` from `class_public/` and is separate from this data-generation workflow. `class_public/doc/input/make1.sh` and `make2.sh` generate documentation and first switch to their own directory.

## Output format

Each sample produces five compressed NPZ files with the default redshifts. For training sample 0:

```text
classpt_sinu_halofit_00000_cmb.npz
classpt_sinu_halofit_00000_derived.npz
classpt_sinu_halofit_00000_z0_pk.npz
classpt_sinu_halofit_00000_z2p5_pk.npz
classpt_sinu_halofit_00000_z5_pk.npz
```

| File | Main fields and units |
| --- | --- |
| `*_cmb.npz` | `ell`, `tt`, `ee`, `te`, `pp`; ell starts at 2; TT/EE/TE are raw dimensionless C_l, and pp is C_l^{phi phi} |
| `*_pk.npz` | `k` (Mpc⁻¹), `z`, `pk_lin` and `pk_nl` (Mpc³) |
| `*_derived.npz` | `names`, `values`, `derived_json` |

Derived parameters include `100*theta_s`, `sigma8`, `YHe`, `z_reio`, `Neff`, `tau_rec`, `z_rec`, `rs_rec`, `ra_rec`, and `rs_d`.

All files contain `meta_json`, which records the sample index, input parameters, redshifts, runtime, extension path, and available version information. Spectrum files also contain `units`. JSON fields are stored as NumPy string scalars:

```python
import json
import numpy as np

path = "output/classpt_sinu_halofit_00000_z0_pk.npz"
with np.load(path, allow_pickle=False) as data:
    k = data["k"]
    pk_lin = data["pk_lin"]
    pk_nl = data["pk_nl"]
    metadata = json.loads(data["meta_json"].item())
    units = json.loads(data["units"].item())
```

CMB outputs are not converted to D_l or μK². To obtain TT/EE/TE D_l in μK², use `ell * (ell + 1) / (2 * np.pi) * C_l * (2.7255e6)**2`. This conversion does not apply to `pp`.

At the default redshifts, 68000 samples produce 340000 result files. Rerunning with the same output directory, prefix, and sample index overwrites matching files. A computation failure terminates the current job; failed samples and existing results are not skipped automatically.

## Validation status and troubleshooting

Completed checks include Python/Shell syntax, LHS stratification and reproducibility, the output workflow with a mock CLASS interface, and Cython translation of the added interface. **A full build of the modified extension and validation with actual spectrum calculations have not yet been completed.**

| Symptom | What to check |
| --- | --- |
| `No module named classy` | Install the local extension in the active Python environment |
| `ADDR2LINE: unbound variable` during Conda activation | Use the updated job scripts with `set -eo pipefail` instead of `set -euo pipefail` |
| Missing `pk_halofit` | In `multinest`, run `make clean` and `make all` from `class_public/`, then verify `hasattr(Class, "pk_halofit")`; the error reports the imported extension and Python executable |
| GCC/OpenBLAS/OpenMP not found during compilation | Check the original cluster paths in Makefile and setup.py |
| Parameter file not found | Run sampling first; pass `--params-file` when using a custom directory |
| Invalid sample index range | Ensure `0 <= start < end <= sample count` and check the fixed job partition sizes |
| Conda or its environment is unavailable | Ensure `conda` is on `PATH` and the `multinest` environment exists |
| Shell not configured for `conda activate` | Use the updated scripts, which source `$(conda info --base)/etc/profile.d/conda.sh` before activation |
| CLASS input or numerical error | Inspect the parameters at the failing index; LHS does not guarantee numerical convergence or model validity across the entire parameter range |

When using this self-interacting neutrino branch for research, follow the citation requirements in the [upstream branch documentation](class_public/README.md).
