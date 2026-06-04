from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import torch
import torchvision.transforms.functional as TF
from diffusers import FluxPipeline
from PIL import Image

from .flowedit_utils import FlowEditFLUX
from .project_utils import (
    case_scout_path,
    copy_source_image,
    filter_records,
    iter_cases,
    load_dataset,
    load_yaml,
    prepare_case_dirs,
)


def _resolve_device(requested: str | None = None) -> str:
    if requested and requested.lower() != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dtype(device: str, requested: str | None = None) -> torch.dtype:
    if requested == "float32":
        return torch.float32
    if requested == "float16":
        return torch.float16
    return torch.float16 if device.startswith("cuda") else torch.float32


def load_flux_pipeline(config: Dict, device: str | None = None, dtype: torch.dtype | None = None) -> FluxPipeline:
    requested_device = _resolve_device(device or config.get("run", {}).get("device"))
    requested_dtype = dtype or _resolve_dtype(requested_device, config.get("run", {}).get("dtype"))
    model_id = config["models"]["flux_model_id"]

    pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=requested_dtype)
    pipe = pipe.to(requested_device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_scout_image_flux(
    pipe: FluxPipeline,
    case: Dict,
    config: Dict,
    image_dir: str | Path,
    case_dir: str | Path,
) -> Path:
    flow_cfg = config["flowedit"]
    source_image = Image.open(case["image_path"]).convert("RGB")
    device = next(pipe.transformer.parameters()).device
    dtype = next(pipe.transformer.parameters()).dtype

    img_tensor = TF.to_tensor(source_image).unsqueeze(0).to(device=device, dtype=dtype) * 2.0 - 1.0
    with torch.no_grad():
        x_src = pipe.vae.encode(img_tensor).latent_dist.sample()
        x_src = (x_src - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

        out_latents = FlowEditFLUX(
            pipe=pipe,
            scheduler=pipe.scheduler,
            x_src=x_src,
            src_prompt=case["source_prompt"],
            tar_prompt=case["target_prompt"],
            negative_prompt=flow_cfg.get("negative_prompt", ""),
            T_steps=int(flow_cfg["T_steps"]),
            n_avg=int(flow_cfg["n_avg"]),
            src_guidance_scale=float(flow_cfg["src_guidance_scale"]),
            tar_guidance_scale=float(flow_cfg["tar_guidance_scale"]),
            n_min=int(flow_cfg["n_min"]),
            n_max=int(flow_cfg["n_max"]),
        )

        image = pipe.vae.decode(
            out_latents / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor,
            return_dict=False,
        )[0]

    scout_image = pipe.image_processor.postprocess(image, output_type="pil")[0]
    image_name = case["image_name"]
    scout_path = case_scout_path(case_dir, case["target_code"])
    scout_image.save(scout_path)
    copy_source_image(case["image_path"], image_dir, image_name)
    return scout_path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FLUX scout images with FlowEdit.")
    parser.add_argument("--config", default="configs/flux.yaml")
    parser.add_argument("--image-name", action="append", dest="image_names")
    parser.add_argument("--target-code", action="append", dest="target_codes")
    args = parser.parse_args()

    config = load_yaml(args.config)
    cases = _collect_cases(config, args.image_names, args.target_codes)
    if not cases:
        raise SystemExit("No FLUX cases matched the current filters.")

    pipe = load_flux_pipeline(config)
    results_root = config["results"]["root"]

    for case in cases:
        image_dir, case_dir = prepare_case_dirs(results_root, case["image_name"], case["target_code"])
        scout_path = generate_scout_image_flux(pipe, case, config, image_dir, case_dir)
        print(f"[flowedit_flux] saved scout for {case['image_name']} / {case['target_code']} -> {scout_path}")


if __name__ == "__main__":
    main()
