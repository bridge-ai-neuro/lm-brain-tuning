from __future__ import annotations

import csv
import inspect
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .compat import (
    filter_voxels,
    normalize_per_runs,
    select_runs,
    tokenize_and_aggregate,
)
from .config import RunConfig
from .data import CanonicalDataset
from .io import atomic_write_json, git_revision, runtime_metadata, sha256_file


def build_training_tensors(
    dataset: CanonicalDataset, config: RunConfig, subject: str, tokenizer: Any
) -> dict[str, Any]:
    if subject not in config.subjects:
        raise ValueError(f"Subject {subject!r} is not part of {config.dataset_key}")
    contexts, chunks, responses = select_runs(
        config.train_folds,
        dataset.contexts,
        dataset.context_chunks,
        dataset.responses,
        dataset.runs,
    )
    responses = normalize_per_runs(responses, dataset.runs, config.train_folds)
    responses = filter_voxels(responses, dataset.combined_masks(config.noise_ceiling_threshold))
    encoded, mapping = tokenize_and_aggregate(
        config.model_id, tokenizer, contexts, chunks, "concatenate_avg_tr"
    )
    encoded["mapping"] = mapping
    encoded["labels_regression"] = np.asarray(responses[subject], dtype=np.float32)
    return encoded


class PaperCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        import torch

        batch = {
            key: torch.stack(
                [value if torch.is_tensor(value) else torch.as_tensor(value) for value in values]
            )
            for key, values in ((key, [row[key] for row in rows]) for key in rows[0])
        }
        labels = batch["input_ids"].clone()
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels
        return batch


def _rows_from_tensors(tensors: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {key: value[index] for key, value in tensors.items()}
        for index in range(len(tensors["input_ids"]))
    ]


def _write_log_history(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def train(
    config: RunConfig,
    *,
    dataset_root: str | Path,
    subject: str,
    output_root: str | Path,
    selected_step: int | None = None,
    selection_metadata: dict[str, object] | None = None,
    evaluation_steps: int | None = None,
) -> Path:
    import torch
    from transformers import AutoTokenizer, Trainer, TrainingArguments

    from .modeling import BrainTuningConfig, BrainTuningModel

    dataset = CanonicalDataset.load(dataset_root)
    dataset.validate_for_run(config, subject)
    dataset_manifest_sha256 = sha256_file(dataset.root / "manifest.json")
    if selected_step is not None and selected_step < 1:
        raise ValueError("selected_step must be positive")
    if selection_metadata is not None:
        from .checkpoints import validate_selection_manifest

        validate_selection_manifest(
            selection_metadata,
            run=config.run_name,
            subject=subject,
            dataset_manifest_sha256=dataset_manifest_sha256,
            selection_training=True,
        )
        if int(selection_metadata["step"]) != selected_step:
            raise ValueError("selected_step and selection manifest step do not match")
    if evaluation_steps is not None and evaluation_steps < 1:
        raise ValueError("evaluation_steps must be positive")
    if evaluation_steps is None and selected_step is None:
        evaluation_steps = config.checkpoint_eval_steps

    output = Path(output_root).resolve() / config.run_name / f"sub-{subject}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty run directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    run_manifest = {
        "status": "initializing",
        "run": config.run_name,
        "subject": subject,
        "train_folds": list(config.train_folds),
        "test_fold": config.test_fold,
        "regime": config.regime,
        "seed": config.seed,
        "seed_scope": "trainer_after_model_initialization",
        "selected_step": selected_step,
        "configuration": asdict(config),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "runtime": runtime_metadata(),
        "repository_revision": git_revision(Path(__file__).resolve().parents[2]),
    }
    atomic_write_json(output / "run.json", run_manifest)

    if config.regime == "pretrained":
        run_manifest["status"] = "no-training-required"
        atomic_write_json(output / "run.json", run_manifest)
        return output

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if "bert" not in config.model_id:
        tokenizer.pad_token = tokenizer.eos_token
    tensors = build_training_tensors(dataset, config, subject, tokenizer)
    rows = _rows_from_tensors(tensors)
    if selected_step is None:
        validation_size = int(len(rows) * config.validation_fraction)
        if validation_size < 1:
            raise ValueError("Validation split is empty")
        train_rows = rows[:-validation_size]
        validation_rows = rows[-validation_size:]
    else:
        train_rows = rows
        validation_rows = None

    model_config = BrainTuningConfig(
        backbone_name=config.model_id,
        num_regression_labels=int(tensors["labels_regression"].shape[1]),
        context_groups=config.context_groups,
        pad_token_id=tokenizer.pad_token_id,
        lm_weight=config.lm_weight,
        brain_weight=config.brain_weight,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=list(config.target_modules),
    )
    model = BrainTuningModel(model_config)

    def compute_metrics(eval_prediction):
        predictions = eval_prediction.predictions
        additional = predictions[-1] if isinstance(predictions, tuple) else predictions
        if not isinstance(additional, dict):
            return {}
        return {
            "lm_loss": float(np.mean(additional["lm_loss"])),
            "regression_loss": float(np.mean(additional["regression_loss"])),
        }

    evaluation_strategy = (
        "no" if selected_step is not None else ("steps" if evaluation_steps else "epoch")
    )
    save_strategy = "steps" if selected_step is not None else evaluation_strategy
    training_arguments = {
        "output_dir": str(output),
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "per_device_train_batch_size": config.batch_size,
        "per_device_eval_batch_size": 16,
        "num_train_epochs": config.epochs,
        "max_steps": -1,
        "eval_strategy": evaluation_strategy,
        "save_strategy": save_strategy,
        "eval_steps": evaluation_steps if selected_step is None else None,
        "save_steps": 1 if selected_step is not None else evaluation_steps,
        "logging_strategy": "epoch",
        "logging_first_step": True,
        "seed": config.seed,
        "report_to": [],
        "remove_unused_columns": False,
        "label_names": ["labels_regression"],
    }
    if "save_safetensors" in inspect.signature(TrainingArguments).parameters:
        training_arguments["save_safetensors"] = False
    arguments = TrainingArguments(**training_arguments)

    class PaperTrainer(Trainer):
        def _save_checkpoint(self, model, trial, *args, **kwargs):
            if selected_step is None or self.state.global_step == selected_step:
                return super()._save_checkpoint(model, trial, *args, **kwargs)
            return None

    trainer = PaperTrainer(
        model=model,
        args=arguments,
        train_dataset=train_rows,
        eval_dataset=validation_rows,
        data_collator=PaperCollator(tokenizer.pad_token_id),
        compute_metrics=compute_metrics if validation_rows is not None else None,
    )
    trainer.train()
    # Paper checkpoints were merged after training.  Keep the same numerical
    # representation while retaining ordinary Transformers checkpoint folders.
    for checkpoint in sorted(output.glob("checkpoint-*")):
        if checkpoint.is_dir():
            BrainTuningModel.from_pretrained(checkpoint).merge_lora().save_pretrained(checkpoint)
    if selected_step is None:
        model.merge_lora()
        trainer.save_model(output / "final")
        tokenizer.save_pretrained(output / "final")
    else:
        selected_checkpoint = output / f"checkpoint-{selected_step}"
        if not selected_checkpoint.is_dir():
            raise RuntimeError(
                f"Selected step {selected_step} was not reached during full-data training"
            )
        tokenizer.save_pretrained(selected_checkpoint)
        atomic_write_json(
            output / "selection.json",
            {
                "step": selected_step,
                "checkpoint_path": str(selected_checkpoint.resolve()),
                "selection_training": False,
                "run": config.run_name,
                "subject": subject,
                "dataset_manifest_sha256": dataset_manifest_sha256,
            },
        )
    _write_log_history(output / "log.csv", trainer.state.log_history)
    run_manifest["status"] = "complete"
    run_manifest["global_step"] = trainer.state.global_step
    if selected_step is not None:
        run_manifest["selected_checkpoint"] = str(selected_checkpoint.resolve())
    run_manifest["runtime"] = runtime_metadata()
    atomic_write_json(output / "run.json", run_manifest)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output
