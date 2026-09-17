from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .io import atomic_write_json, sha256_file


def validate_selection_manifest(
    selection: dict[str, object],
    *,
    run: str,
    subject: str,
    dataset_manifest_sha256: str,
    selection_training: bool,
) -> None:
    required = {"step", "run", "subject", "dataset_manifest_sha256", "selection_training"}
    if not selection_training:
        required.add("checkpoint_path")
    missing = sorted(required.difference(selection))
    if missing:
        raise ValueError(f"Selection manifest is missing provenance fields: {missing}")
    expected = {
        "run": run,
        "subject": subject,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "selection_training": selection_training,
    }
    mismatches = {
        field: (selection[field], value)
        for field, value in expected.items()
        if selection[field] != value
    }
    if mismatches:
        raise ValueError(f"Selection manifest does not match this run: {mismatches}")
    if int(selection["step"]) < 1:
        raise ValueError("Selection manifest step must be positive")


def select_checkpoint(
    log_path: str | Path,
    output: str | Path,
    checkpoint_root: str | Path | None = None,
) -> dict[str, float | int | str]:
    """Replicate the normalized LM-plus-regression validation criterion."""
    log_path = Path(log_path).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint selection: {output}")
    with log_path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle)]
    valid = [
        row
        for row in rows
        if row.get("eval_lm_loss") not in (None, "")
        and row.get("eval_regression_loss") not in (None, "")
    ]
    if not valid:
        raise ValueError("No rows contain both validation losses")
    language = np.asarray([float(row["eval_lm_loss"]) for row in valid])
    regression = np.asarray([float(row["eval_regression_loss"]) for row in valid])

    def minmax(values: np.ndarray) -> np.ndarray:
        return (values - values.min()) / (values.max() - values.min())

    combined = minmax(language) + minmax(regression)
    best = valid[int(np.nanargmin(combined))]
    selected = {
        "step": int(float(best["step"])),
        "epoch": float(best["epoch"]),
        "eval_lm_loss": float(best["eval_lm_loss"]),
        "eval_regression_loss": float(best["eval_regression_loss"]),
        "combined_normalized_loss": float(np.nanmin(combined)),
        "source_log": str(log_path),
        "source_log_sha256": sha256_file(log_path),
    }
    source_manifest_path = log_path.parent / "run.json"
    if source_manifest_path.is_file():
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest.get("status") != "complete":
            raise ValueError(f"Selection training is not complete: {source_manifest_path}")
        for field in ("run", "subject", "dataset_manifest_sha256"):
            if field in source_manifest:
                selected[field] = source_manifest[field]
        selected["selection_training"] = True
    if checkpoint_root is not None:
        checkpoint = Path(checkpoint_root).resolve() / f"checkpoint-{selected['step']}"
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Selected checkpoint does not exist: {checkpoint}")
        selected["checkpoint_path"] = str(checkpoint)
    atomic_write_json(output, selected)
    return selected
