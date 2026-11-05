"""Check call_class.py CMB/derived outputs using ranges from the job scripts."""
import argparse
import re
import shlex
from pathlib import Path

import numpy as np

from call_class import DATA_PARAMETER_MAP, PARAMS, PROJECT_DIR
from sample_status import sample_complete


class JobArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def job_options(path):
    parser = JobArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int)
    parser.add_argument("--params-file", default="dataset/neff_train_param.npz")
    parser.add_argument("--out-dir", default="output")
    parser.add_argument("--prefix", default="classpt_sinu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--index-offset", type=int, default=0)
    parser.add_argument("--pk-z", type=float, nargs="+", default=[0.0])
    commands = []
    for line in path.read_text().replace("\\\n", " ").splitlines():
        tokens = shlex.split(line, comments=True)
        if not tokens or not re.fullmatch(r"python(?:\d+(?:\.\d+)?)?", Path(tokens[0]).name):
            continue
        for position, token in enumerate(tokens):
            if Path(token).name in ("call_class.py", "call_class_pk.py"):
                commands.append((Path(token).name, tokens[position + 1:]))
    if len(commands) != 1:
        raise ValueError("expected one call_class.py or call_class_pk.py command")
    driver, command = commands[0]
    options, unknown = parser.parse_known_args(command)
    if unknown:
        raise ValueError(f"unsupported command arguments: {unknown}")
    options.driver = driver
    if any("$" in value for value in (options.params_file, options.out_dir, options.prefix)):
        raise ValueError("use literal paths/prefix in the Python command")
    return options


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scripts", nargs="*", type=Path,
                        help="Default: scripts/data[0-9]*.sh, scripts/test_data*.sh, scripts/val_data*.sh")
    parser.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    parser.add_argument("--kind", choices=("cmb",), default="cmb",
                        help="Check call_class.py CMB/derived outputs (default: cmb)")
    parser.add_argument("--out-dir", type=Path,
                        help="Override all jobs' output directory, relative to project root")
    args = parser.parse_args()
    root = args.project_dir.resolve()
    print(f"Project: {root}; output type: {args.kind}; "
          f"output directory: {args.out_dir if args.out_dir is not None else 'from each script'}", flush=True)
    scripts = args.scripts or sorted(
        [p for p in (root / "scripts").glob("*.sh")
         if re.fullmatch(r"(?:test_|val_)?data\d+\.sh", p.name)],
        key=lambda p: (p.name.split("data")[0], int(re.search(r"\d+", p.name)[0])),
    )
    print(f"{'Script':<20} {'Range':<16} {'Total':>7} {'Done':>7} {'Pending':>8} {'First pending':>14}", flush=True)
    totals = [0, 0]
    errors = 0
    cache = {}
    for script in scripts:
        try:
            options = job_options(script)
            if args.out_dir is not None:
                options.out_dir = str(args.out_dir)
            print(f"Checking {script.name}: "
                  f"CMB/derived in "
                  f"{root / options.out_dir}", flush=True)
            params_path = root / options.params_file
            if params_path not in cache:
                with np.load(params_path, allow_pickle=False) as archive:
                    cache[params_path] = {key: archive[key] for key in DATA_PARAMETER_MAP}
            data = cache[params_path]
            size = len(data["h"])
            if any(v.ndim != 1 or len(v) != size or not np.isfinite(v).all()
                   for v in data.values()):
                raise ValueError("invalid parameter arrays")
            end = size if options.end is None else options.end
            if not 0 <= options.start < end <= size:
                raise ValueError(f"invalid index range for {size} samples")
            done = 0
            first = "-"
            for index in range(options.start, end):
                output_index = index + options.index_offset
                params = dict(PARAMS)
                params.update({target: data[source][index].item()
                               for source, target in DATA_PARAMETER_MAP.items()})
                complete = sample_complete(root / options.out_dir, options.prefix, output_index, params)
                if complete:
                    done += 1
                elif first == "-":
                    first = str(output_index)
            total = end - options.start
            totals[0] += total
            totals[1] += done
            print(f"{script.name:<20} {f'[{options.start},{end})':<16} {total:>7} {done:>7} {total-done:>8} {first:>14}", flush=True)
        except (OSError, ValueError, KeyError, argparse.ArgumentError) as exc:
            errors += 1
            print(f"{script.name}: ERROR: {exc}", flush=True)
    print(f"Job totals: expected={totals[0]}, done={totals[1]}, pending={totals[0]-totals[1]}, errors={errors}")
    if errors or not scripts:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
