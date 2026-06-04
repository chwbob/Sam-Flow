from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def save_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def resolve_path(base: str | Path, value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return Path(base) / candidate


def image_name_from_record(record: Dict[str, Any]) -> str:
    return Path(record["init_img"]).stem


def _normalize_per_target_list(
    values: List[Any],
    target_count: int,
    field_name: str,
    image_name: str,
    warnings: List[str],
    strict: bool,
) -> List[Any]:
    if len(values) == target_count:
        return values
    if len(values) == 1 and target_count > 1:
        warnings.append(
            f"{image_name}: expanded {field_name} from length 1 to match {target_count} targets."
        )
        return values * target_count
    if len(values) > target_count:
        message = (
            f"{image_name}: truncated {field_name} from length {len(values)} to {target_count} targets."
        )
        if strict:
            raise ValueError(message)
        warnings.append(message)
        return values[:target_count]
    raise ValueError(
        f"{image_name}: {field_name} has length {len(values)} but target_count is {target_count}."
    )


def normalize_record(record: Dict[str, Any], strict: bool = False) -> Tuple[Dict[str, Any], List[str]]:
    image_name = image_name_from_record(record)
    warnings: List[str] = []

    target_prompts = list(record["target_prompts"])
    target_codes = list(record["target_codes"])
    target_count = len(target_prompts)

    if target_count != len(target_codes):
        raise ValueError(
            f"{image_name}: target_prompts has length {target_count} but target_codes has length {len(target_codes)}."
        )

    normalized = dict(record)
    normalized["target_prompts"] = target_prompts
    normalized["target_codes"] = target_codes
    normalized["source_mask_token"] = _normalize_per_target_list(
        list(record["source_mask_token"]),
        target_count,
        "source_mask_token",
        image_name,
        warnings,
        strict,
    )
    normalized["target_mask_token"] = _normalize_per_target_list(
        list(record["target_mask_token"]),
        target_count,
        "target_mask_token",
        image_name,
        warnings,
        strict,
    )
    normalized["unchanged_token"] = _normalize_per_target_list(
        list(record["unchanged_token"]),
        target_count,
        "unchanged_token",
        image_name,
        warnings,
        strict,
    )
    return normalized, warnings


def load_dataset(dataset_path: str | Path, strict: bool = False) -> Tuple[List[Dict[str, Any]], List[str]]:
    dataset = load_yaml(dataset_path)
    if not isinstance(dataset, list):
        raise ValueError(f"Dataset file {dataset_path} must contain a top-level list.")

    normalized_records: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for index, record in enumerate(dataset):
        if not isinstance(record, dict):
            raise ValueError(f"Dataset item {index} is not a mapping.")
        normalized, item_warnings = normalize_record(record, strict=strict)
        normalized["_record_index"] = index
        normalized_records.append(normalized)
        warnings.extend(item_warnings)
    return normalized_records, warnings


def filter_records(
    records: Iterable[Dict[str, Any]],
    image_names: Iterable[str] | None = None,
    start_index: int | None = None,
    end_index: int | None = None,
    max_records: int | None = None,
) -> List[Dict[str, Any]]:
    wanted = {name.strip() for name in image_names or [] if name and name.strip()}
    selected: List[Dict[str, Any]] = []
    for record in records:
        record_index = int(record["_record_index"])
        if start_index is not None and record_index < start_index:
            continue
        if end_index is not None and record_index > end_index:
            continue
        if wanted and image_name_from_record(record) not in wanted:
            continue
        selected.append(record)
        if max_records is not None and len(selected) >= max_records:
            break
    return selected


def iter_cases(
    records: Iterable[Dict[str, Any]],
    data_root: str | Path,
    target_codes: Iterable[str] | None = None,
    max_cases: int | None = None,
) -> List[Dict[str, Any]]:
    wanted_codes = {code.strip() for code in target_codes or [] if code and code.strip()}
    cases: List[Dict[str, Any]] = []
    data_root_path = Path(data_root)

    for record in records:
        image_name = image_name_from_record(record)
        image_path = resolve_path(data_root_path, record["init_img"])
        for target_index, target_prompt in enumerate(record["target_prompts"]):
            target_code = record["target_codes"][target_index]
            if wanted_codes and target_code not in wanted_codes:
                continue

            case = {
                "record_index": int(record["_record_index"]),
                "target_index": target_index,
                "image_name": image_name,
                "image_path": image_path,
                "source_prompt": record["source_prompt"],
                "target_prompt": target_prompt,
                "target_code": target_code,
                "source_mask_tokens": list(record["source_mask_token"][target_index]),
                "target_mask_tokens": list(record["target_mask_token"][target_index]),
                "unchanged_tokens": list(record["unchanged_token"][target_index]),
            }
            cases.append(case)
            if max_cases is not None and len(cases) >= max_cases:
                return cases
    return cases


def make_single_case(
    image_path: str | Path,
    source_prompt: str,
    target_prompt: str,
    source_mask_tokens: Iterable[str] | None = None,
    target_mask_tokens: Iterable[str] | None = None,
    unchanged_tokens: Iterable[str] | None = None,
    target_code: str = "edit",
) -> Dict[str, Any]:
    resolved_image_path = Path(image_path)
    return {
        "record_index": 0,
        "target_index": 0,
        "image_name": resolved_image_path.stem,
        "image_path": resolved_image_path,
        "source_prompt": source_prompt,
        "target_prompt": target_prompt,
        "target_code": target_code,
        "source_mask_tokens": list(source_mask_tokens or []),
        "target_mask_tokens": list(target_mask_tokens or []),
        "unchanged_tokens": list(unchanged_tokens or []),
    }


def prepare_case_dirs(
    results_root: str | Path,
    image_name: str,
    target_code: str,
) -> Tuple[Path, Path]:
    image_dir = ensure_dir(Path(results_root) / image_name)
    case_dir = ensure_dir(image_dir / target_code)
    return image_dir, case_dir


def copy_source_image(image_path: str | Path, image_dir: str | Path, image_name: str) -> Path:
    source_path = Path(image_path)
    destination = Path(image_dir) / f"{image_name}{source_path.suffix or '.png'}"
    if not destination.exists():
        shutil.copy2(source_path, destination)
    return destination


def case_output_path(case_dir: str | Path, target_code: str) -> Path:
    return Path(case_dir) / f"{target_code}_output.png"


def case_scout_path(case_dir: str | Path, target_code: str) -> Path:
    return Path(case_dir) / f"{target_code}_scout.png"


def save_case_manifest(
    case_dir: str | Path,
    case: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Path:
    payload = {
        "record_index": case["record_index"],
        "target_index": case["target_index"],
        "image_name": case["image_name"],
        "image_path": str(case["image_path"]),
        "source_prompt": case["source_prompt"],
        "target_prompt": case["target_prompt"],
        "target_code": case["target_code"],
        "source_mask_tokens": case["source_mask_tokens"],
        "target_mask_tokens": case["target_mask_tokens"],
        "unchanged_tokens": case["unchanged_tokens"],
    }
    if extra:
        payload.update(extra)
    manifest_path = Path(case_dir) / "case.yaml"
    save_yaml(manifest_path, payload)
    return manifest_path
