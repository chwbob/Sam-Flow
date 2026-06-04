from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam_flow.project_utils import (
    case_scout_path,
    case_output_path,
    filter_records,
    iter_cases,
    load_dataset,
    load_yaml,
    prepare_case_dirs,
)


def _config_path_for_mode(mode: str) -> str:
    return "configs/flux.yaml" if mode == "flux" else "configs/sd3.yaml"


def _collect_cases(config: Dict, image_names: List[str] | None, target_codes: List[str] | None) -> List[Dict]:
    dataset_path = config["dataset"]["path"]
    data_root = config["dataset"]["root"]
    strict = bool(config["dataset"].get("strict_validation", False))
    records, warnings = load_dataset(dataset_path, strict=strict)
    for warning in warnings:
        print(f"[dataset warning] {warning}")

    selected_records = filter_records(
        records,
        image_names=image_names or config.get("run", {}).get("image_names"),
        start_index=config.get("run", {}).get("start_index"),
        end_index=config.get("run", {}).get("end_index"),
        max_records=config.get("run", {}).get("max_records"),
    )
    return iter_cases(
        selected_records,
        data_root=data_root,
        target_codes=target_codes or config.get("run", {}).get("target_codes"),
        max_cases=config.get("run", {}).get("max_cases"),
    )


def _run_flux(config: Dict, cases: List[Dict]) -> None:
    from sam_flow.flowedit_flux import generate_scout_image_flux, load_flux_pipeline
    from sam_flow.sam_flow_flux import run_sam_flow_flux_case

    pipe = load_flux_pipeline(config)
    results_root = config["results"]["root"]
    overwrite = bool(config.get("run", {}).get("overwrite", False))

    for case in cases:
        image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
        out_path = case_output_path(case_dir, case["target_code"])
        if out_path.exists() and not overwrite:
            print(f"[run] skip existing FLUX output {out_path}")
            continue

        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_flux(pipe, case, config, image_dir, case_dir)

        out_path = run_sam_flow_flux_case(pipe, case, config, scout_path, case_dir)
        print(f"[run] FLUX completed {case['image_name']} / {case['target_code']} -> {out_path}")


def _run_sd3(config: Dict, cases: List[Dict]) -> None:
    from sam_flow.flowedit_sd3 import generate_scout_image_sd3, load_sd3_pipeline
    from sam_flow.sam_flow_sd3 import run_sam_flow_sd3_case

    pipe = load_sd3_pipeline(config)
    results_root = config["results"]["root"]
    overwrite = bool(config.get("run", {}).get("overwrite", False))

    for case in cases:
        image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
        out_path = case_output_path(case_dir, case["target_code"])
        if out_path.exists() and not overwrite:
            print(f"[run] skip existing SD3 output {out_path}")
            continue

        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_sd3(pipe, case, config, image_dir, case_dir)

        out_path = run_sam_flow_sd3_case(pipe, case, config, scout_path, case_dir)
        print(f"[run] SD3 completed {case['image_name']} / {case['target_code']} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM-Flow on a dataset config.")
    parser.add_argument("--mode", choices=["flux", "sd3"])
    parser.add_argument("--config")
    parser.add_argument("--image-name", action="append", dest="image_names")
    parser.add_argument("--target-code", action="append", dest="target_codes")
    args = parser.parse_args()

    if args.config:
        config = load_yaml(args.config)
        mode = args.mode or config.get("mode")
        if mode not in {"flux", "sd3"}:
            raise SystemExit("Mode must be set via --mode or the config file.")
    else:
        mode = args.mode or "flux"
        config = load_yaml(_config_path_for_mode(mode))

    cases = _collect_cases(config, args.image_names, args.target_codes)
    if not cases:
        raise SystemExit("No cases matched the current filters.")

    if mode == "flux":
        _run_flux(config, cases)
    else:
        _run_sd3(config, cases)


if __name__ == "__main__":
    main()
