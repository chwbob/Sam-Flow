from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sam_flow.project_utils import (
    case_output_path,
    case_scout_path,
    load_yaml,
    make_single_case,
    prepare_case_dirs,
)


def _split_tokens(values: list[str] | None) -> list[str]:
    tokens: list[str] = []
    for value in values or []:
        tokens.extend(part.strip() for part in value.split(",") if part.strip())
    return tokens


def _default_config(mode: str) -> str:
    return "configs/flux.yaml" if mode == "flux" else "configs/sd3.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run SAM-Flow on one image with custom prompts.")
    parser.add_argument("--mode", choices=["flux", "sd3"], default="flux")
    parser.add_argument("--config", help="Path to a SAM-Flow YAML config.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--source-prompt", required=True, help="Prompt describing the input image.")
    parser.add_argument("--target-prompt", required=True, help="Prompt describing the desired edit.")
    parser.add_argument("--source-token", action="append", dest="source_tokens", help="Source object token. Repeat or comma-separate.")
    parser.add_argument("--target-token", action="append", dest="target_tokens", help="Target object token. Repeat or comma-separate.")
    parser.add_argument("--unchanged-token", action="append", dest="unchanged_tokens", help="Token to preserve. Repeat or comma-separate.")
    parser.add_argument("--target-code", default="edit", help="Name used for the output subdirectory and file prefix.")
    parser.add_argument("--output-root", help="Override the results root from the config.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate scout and final output if they already exist.")
    args = parser.parse_args()

    config = load_yaml(args.config or _default_config(args.mode))
    if args.output_root:
        config.setdefault("results", {})["root"] = args.output_root
    if args.overwrite:
        config.setdefault("run", {})["overwrite"] = True

    source_tokens = _split_tokens(args.source_tokens)
    target_tokens = _split_tokens(args.target_tokens)
    unchanged_tokens = _split_tokens(args.unchanged_tokens)
    if not unchanged_tokens and (not source_tokens or not target_tokens):
        raise SystemExit("Provide --source-token and --target-token, or provide --unchanged-token.")

    case = make_single_case(
        image_path=args.image,
        source_prompt=args.source_prompt,
        target_prompt=args.target_prompt,
        source_mask_tokens=source_tokens,
        target_mask_tokens=target_tokens,
        unchanged_tokens=unchanged_tokens,
        target_code=args.target_code,
    )

    results_root = config["results"]["root"]
    overwrite = bool(config.get("run", {}).get("overwrite", False))
    image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
    out_path = case_output_path(case_dir, case["target_code"])

    if out_path.exists() and not overwrite:
        print(f"[run_image] skip existing output {out_path}")
        return

    if args.mode == "flux":
        from sam_flow.flowedit_flux import generate_scout_image_flux, load_flux_pipeline
        from sam_flow.sam_flow_flux import run_sam_flow_flux_case

        pipe = load_flux_pipeline(config)
        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_flux(pipe, case, config, image_dir, case_dir)
        out_path = run_sam_flow_flux_case(pipe, case, config, scout_path, case_dir)
    else:
        from sam_flow.flowedit_sd3 import generate_scout_image_sd3, load_sd3_pipeline
        from sam_flow.sam_flow_sd3 import run_sam_flow_sd3_case

        pipe = load_sd3_pipeline(config)
        scout_path = case_scout_path(case_dir, case["target_code"])
        if overwrite or not scout_path.exists():
            scout_path = generate_scout_image_sd3(pipe, case, config, image_dir, case_dir)
        out_path = run_sam_flow_sd3_case(pipe, case, config, scout_path, case_dir)

    print(f"[run_image] saved {out_path}")


if __name__ == "__main__":
    main()
