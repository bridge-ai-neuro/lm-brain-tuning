from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from .compat import (
    concatenate_average_tr,
    correlation_by_target,
    normalize_all,
    normalize_per_runs,
    select_runs,
    tokenize_and_aggregate,
)
from .config import RunConfig
from .data import CanonicalDataset
from .io import atomic_write_json, runtime_metadata, sha256_file
from .ridge import fit_cpu_ridge, fit_torch_ridge


def _load_backbone(model_path: str, model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from .modeling import BrainTuningModel

    path = Path(model_path)
    tokenizer_source = model_path if (path / "tokenizer_config.json").exists() else model_id
    if path.exists() and (path / "config.json").exists():
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") == "lm_brain_tuning":
            wrapper = BrainTuningModel.from_pretrained(path)
            return wrapper.backbone_lm, AutoTokenizer.from_pretrained(tokenizer_source)
    return (
        AutoModelForCausalLM.from_pretrained(model_path),
        AutoTokenizer.from_pretrained(tokenizer_source),
    )


def _extract_features(
    model: Any,
    encoded: dict[str, Any],
    mapping: Any,
    *,
    batch_size: int,
    compute_loss: bool,
    pad_token_id: int,
) -> tuple[Any, list[float]]:
    """Extract batched features and optional language-model losses."""
    import torch

    features = []
    losses: list[float] = []
    sample_count = len(encoded["input_ids"])
    with torch.no_grad():
        for offset in range(0, sample_count, batch_size):
            stop = min(offset + batch_size, sample_count)
            input_ids = encoded["input_ids"][offset:stop].to(model.device)
            attention_mask = encoded["attention_mask"][offset:stop].to(model.device)
            labels = None
            if compute_loss:
                labels = input_ids.clone()
                labels[labels == pad_token_id] = -100
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                output_hidden_states=True,
            )
            if compute_loss:
                losses.append(float(output.loss.detach().cpu()))
            paper_mapping = mapping[: stop - offset]
            features.append(
                concatenate_average_tr(output.hidden_states[-1], paper_mapping).detach().cpu()
            )
    return torch.cat(features), losses


def _torch_zscore(values):
    import torch

    mean = torch.mean(values, dim=0, keepdim=True)
    variance = torch.var(values, dim=0, keepdim=True, correction=False)
    return (values - mean) / torch.sqrt(variance)


def _torch_zscore_per_runs(values, original_runs, selected_runs):
    import torch

    runs = np.asarray(original_runs)
    filtered_runs = runs[np.isin(runs, selected_runs)]
    groups = [_torch_zscore(values[filtered_runs == run]) for run in selected_runs]
    return torch.cat(groups, dim=0)


def evaluate_brain(
    config: RunConfig,
    *,
    dataset_root: str | Path,
    subject: str,
    model_path: str,
    output_root: str | Path,
    device: str = "cuda",
    feature_batch_size: int = 8,
    selection_metadata: dict[str, object] | None = None,
) -> Path:
    import torch

    if feature_batch_size < 1:
        raise ValueError("feature_batch_size must be positive")
    dataset = CanonicalDataset.load(dataset_root)
    dataset.validate_for_run(config, subject)
    dataset_manifest_sha256 = sha256_file(dataset.root / "manifest.json")
    if selection_metadata is not None:
        from .checkpoints import validate_selection_manifest

        validate_selection_manifest(
            selection_metadata,
            run=config.run_name,
            subject=subject,
            dataset_manifest_sha256=dataset_manifest_sha256,
            selection_training=False,
        )
        selected_checkpoint = Path(str(selection_metadata["checkpoint_path"])).resolve()
        if Path(model_path).resolve() != selected_checkpoint:
            raise ValueError("model_path and selection manifest checkpoint_path do not match")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "The paper evaluation profile requires CUDA; use --device cpu for smoke tests"
        )
    train_sample_count = int(np.isin(dataset.runs, config.train_folds).sum())
    if train_sample_count < config.ridge_splits:
        raise ValueError(
            f"Ridge evaluation needs at least {config.ridge_splits} training samples, "
            f"got {train_sample_count}"
        )

    output = Path(output_root).resolve() / config.run_name / f"sub-{subject}"
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty evaluation: {output}")
    mask = dataset.combined_masks(config.noise_ceiling_threshold)[subject]

    model, tokenizer = _load_backbone(model_path, config.model_id)
    if "bert" not in config.model_id:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    model.eval()

    train_contexts, train_chunks, train_responses = select_runs(
        config.train_folds,
        dataset.contexts,
        dataset.context_chunks,
        dataset.responses,
        dataset.runs,
    )
    test_contexts, test_chunks, test_responses = select_runs(
        (config.test_fold,),
        dataset.contexts,
        dataset.context_chunks,
        dataset.responses,
        dataset.runs,
    )
    train_responses = normalize_per_runs(train_responses, dataset.runs, config.train_folds)
    test_responses = normalize_all(test_responses)

    train_encoded, train_mapping = tokenize_and_aggregate(
        config.model_id, tokenizer, train_contexts, train_chunks, "concatenate_avg_tr"
    )
    test_encoded, test_mapping = tokenize_and_aggregate(
        config.model_id, tokenizer, test_contexts, test_chunks, "concatenate_avg_tr"
    )
    train_features, _ = _extract_features(
        model,
        train_encoded,
        train_mapping,
        batch_size=feature_batch_size,
        compute_loss=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    test_features, test_losses = _extract_features(
        model,
        test_encoded,
        test_mapping,
        batch_size=feature_batch_size,
        compute_loss=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    train_features = _torch_zscore_per_runs(
        train_features.float(), dataset.runs, config.train_folds
    )
    test_features = _torch_zscore(test_features.float())
    train_target = torch.as_tensor(train_responses[subject], dtype=torch.float32)
    test_target = torch.as_tensor(test_responses[subject], dtype=torch.float32)

    if device == "cuda":
        train_features = train_features.to(device)
        test_features = test_features.to(device)
        train_target = train_target.to(device)
        weights = fit_torch_ridge(
            train_features,
            train_target,
            lambdas=config.ridge_lambdas,
            splits=config.ridge_splits,
        )
        prediction = (test_features @ weights).cpu().numpy()
    else:
        weights = fit_cpu_ridge(
            train_features.numpy(),
            train_target.numpy(),
            lambdas=config.ridge_lambdas,
            splits=config.ridge_splits,
        )
        prediction = test_features.numpy() @ weights
    target = test_target.cpu().numpy()
    correlations = correlation_by_target(prediction, target)

    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "predictions.npy", prediction, allow_pickle=False)
    np.save(output / "targets.npy", target, allow_pickle=False)
    np.save(output / "correlations.npy", correlations, allow_pickle=False)
    metrics = {
        "run": config.run_name,
        "subject": subject,
        "model_path": model_path,
        "configuration": asdict(config),
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "lm_loss": float(np.mean(test_losses)),
        "median_language_correlation": float(np.median(correlations[mask])),
        "mean_language_correlation": float(np.mean(correlations[mask])),
        "selected_voxels": int(mask.sum()),
        "runtime": runtime_metadata(),
    }
    atomic_write_json(output / "metrics.json", metrics)
    return output
