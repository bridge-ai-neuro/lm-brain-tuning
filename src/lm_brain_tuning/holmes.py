from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from .config import SUPPORTED_REGIMES
from .io import atomic_write_json, runtime_metadata

PAPER_HOLMES_REVISION = "bded82c8e8941fbc651c2eee7203b01f6a6c1de6"


def _task_subfields(path: str | Path) -> dict[str, str]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        task_field = next(
            (field for field in ("probing_dataset", "probing dataset", "task") if field in fields),
            None,
        )
        subfield_field = next(
            (
                field
                for field in ("linguistic subfield", "linguistic_subfield", "subfield")
                if field in fields
            ),
            None,
        )
        if task_field is None or subfield_field is None:
            raise ValueError("Task metadata must contain task and linguistic-subfield columns")
        mapping: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            task = row[task_field].strip()
            subfield = row[subfield_field].strip()
            if not task or not subfield:
                raise ValueError(f"Invalid task metadata row {row_number}")
            if task in mapping and mapping[task] != subfield:
                raise ValueError(f"Task {task!r} has conflicting linguistic subfields")
            mapping[task] = subfield
    return mapping


def checkout_revision(checkout: str | Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def export_backbone(model_path: str | Path, output: str | Path) -> Path:
    from transformers import AutoTokenizer

    from .modeling import BrainTuningModel

    model_path = Path(model_path)
    output = Path(output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite exported model: {output}")
    wrapper = BrainTuningModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    backbone = wrapper.backbone_lm
    if hasattr(backbone, "merge_and_unload"):
        backbone = backbone.merge_and_unload()
    if "bert" in wrapper.config.backbone_name:
        transformer = backbone.bert
    elif "gpt2" in wrapper.config.backbone_name:
        transformer = backbone.transformer
    else:
        raise ValueError("Only BERT and GPT-2 are part of the paper profile")
    output.mkdir(parents=True, exist_ok=True)
    transformer.save_pretrained(output)
    tokenizer.save_pretrained(output)
    return output


def run_holmes(
    *,
    checkout: str | Path,
    model_path: str | Path,
    output_dir: str | Path,
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4),
    entrypoint: str = "src/investigate.py",
    version: str = "flash-holmes",
) -> Path:
    """Invoke the pinned external checkout without vendoring its source."""
    if tuple(seeds) != (0, 1, 2, 3, 4):
        raise ValueError("The paper Holmes profile requires seeds 0, 1, 2, 3, 4")
    if version != "flash-holmes":
        raise ValueError("The paper Holmes profile requires version 'flash-holmes'")
    checkout = Path(checkout).resolve()
    entrypoint_path = checkout / entrypoint
    if not entrypoint_path.exists():
        raise FileNotFoundError(f"Holmes entrypoint not found: {entrypoint_path}")
    revision = checkout_revision(checkout)
    if revision != PAPER_HOLMES_REVISION:
        raise ValueError(
            "Holmes checkout revision does not match the paper profile: "
            f"expected {PAPER_HOLMES_REVISION}, got {revision}"
        )
    output_dir = Path(output_dir).resolve()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty Holmes output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_source = Path(model_path)
    if model_source.is_dir():
        config_path = model_source / "config.json"
        is_wrapper = (
            config_path.is_file()
            and json.loads(config_path.read_text(encoding="utf-8")).get("model_type")
            == "lm_brain_tuning"
        )
        holmes_model = (
            export_backbone(model_source, output_dir / "exported-model")
            if is_wrapper
            else model_source.resolve()
        )
    else:
        holmes_model = str(model_path)
    command = [
        sys.executable,
        entrypoint_path.name,
        "--model_name",
        str(holmes_model),
        "--version",
        version,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--dump_folder",
        str(output_dir / "dumps"),
        "--result_folder",
        str(output_dir / "results"),
    ]
    manifest = {
        "status": "initializing",
        "checkout": str(checkout),
        "revision": revision,
        "model_path": str(model_path),
        "holmes_model": str(holmes_model),
        "seeds": list(seeds),
        "version": version,
        "command": command,
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output_dir / "run.json", manifest)
    subprocess.run(command, cwd=entrypoint_path.parent, check=True)
    manifest["status"] = "complete"
    manifest["runtime"] = runtime_metadata()
    atomic_write_json(output_dir / "run.json", manifest)
    return output_dir


def normalize_holmes_csv(
    input_path: str | Path,
    output_path: str | Path,
    system: str,
    comparison_unit: str,
    analysis_unit: str,
    aggregation_unit: str,
    task_metadata: str | Path | None = None,
) -> Path:
    """Normalize a FlashHolmes CSV into the public result schema."""
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite normalized Holmes CSV: {output_path}")
    identities = {
        "system": system.strip(),
        "comparison_unit": comparison_unit.strip(),
        "analysis_unit": analysis_unit.strip(),
        "aggregation_unit": aggregation_unit.strip(),
    }
    if any(not value for value in identities.values()):
        raise ValueError("Holmes normalization identity fields must not be empty")
    if identities["system"] not in SUPPORTED_REGIMES:
        raise ValueError(f"system must be one of {SUPPORTED_REGIMES}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Path(input_path).open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
    required = {"probing_dataset", "model_name", "seed", "score"}
    if not rows or not required.issubset(reader.fieldnames or []):
        raise ValueError(f"Holmes CSV must contain {sorted(required)}")
    source_subfield_field = next(
        (
            field
            for field in ("linguistic subfield", "linguistic_subfield", "subfield")
            if field in (reader.fieldnames or [])
        ),
        None,
    )
    metadata = _task_subfields(task_metadata) if task_metadata is not None else {}
    normalized = []
    for row_number, row in enumerate(rows, start=2):
        try:
            seed = int(row["seed"])
            score = float(row["score"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid Holmes row {row_number}") from error
        task = row["probing_dataset"]
        source_subfield = (
            row[source_subfield_field].strip() if source_subfield_field is not None else ""
        )
        subfield = source_subfield or metadata.get(task, "")
        if not task or not subfield or not row["model_name"] or not math.isfinite(score):
            raise ValueError(f"Invalid Holmes row {row_number}")
        normalized.append(
            {
                "task": task,
                "subfield": subfield,
                **identities,
                "model_name": row["model_name"],
                "seed": seed,
                "score": score,
            }
        )
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=[
                "task",
                "subfield",
                "system",
                "comparison_unit",
                "analysis_unit",
                "aggregation_unit",
                "model_name",
                "seed",
                "score",
            ],
        )
        writer.writeheader()
        writer.writerows(normalized)
    os.replace(temporary, output_path)
    return output_path
