from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUPPORTED_DATASETS = ("hp", "moth")
SUPPORTED_MODELS = ("bert", "gpt2")
SUPPORTED_REGIMES = ("pretrained", "brain", "stimulus", "joint")
EXPECTED_SUBJECTS = {
    "hp": ("F", "H", "I", "J", "K", "L", "M", "N"),
    "moth": ("01", "02", "03", "05", "07", "08"),
}


def load_mapping(path: str | Path) -> dict[str, Any]:
    """Load JSON or YAML without making YAML mandatory for lightweight checks."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        value = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            value = json.loads(text)
        else:
            value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must contain a mapping: {path}")
    return value


@dataclass(frozen=True)
class RunConfig:
    model_key: str
    model_id: str
    dataset_key: str
    subjects: tuple[str, ...]
    regime: str
    test_fold: int
    folds: int
    seed: int
    epochs: int
    batch_size: int
    validation_fraction: float
    context_words: int
    context_groups: int
    checkpoint_eval_steps: int
    noise_ceiling_threshold: float
    learning_rate: float
    weight_decay: float
    fmri_aggregation: str
    lm_weight: float
    brain_weight: float
    lora_rank: int | None
    lora_alpha: int | None
    lora_dropout: float
    target_modules: tuple[str, ...]
    ridge_lambdas: tuple[float, ...]
    ridge_splits: int
    holmes_seeds: tuple[int, ...]

    @property
    def train_folds(self) -> tuple[int, ...]:
        return tuple(fold for fold in range(self.folds) if fold != self.test_fold)

    @property
    def run_name(self) -> str:
        return f"{self.model_key}-{self.dataset_key}-{self.regime}-fold{self.test_fold}"


def resolve_run_config(
    path: str | Path,
    *,
    model: str,
    dataset: str,
    regime: str,
    test_fold: int,
    seed: int | None = None,
) -> RunConfig:
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model {model!r}; choose from {SUPPORTED_MODELS}")
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset!r}; choose from {SUPPORTED_DATASETS}")
    if regime not in SUPPORTED_REGIMES:
        raise ValueError(f"Unsupported regime {regime!r}; choose from {SUPPORTED_REGIMES}")

    raw = load_mapping(path)
    folds = int(raw["experiment"]["folds"])
    if test_fold not in range(folds):
        raise ValueError(f"test_fold must be between 0 and {folds - 1}")
    model_cfg = raw["models"][model]
    dataset_cfg = raw["datasets"][dataset]
    regime_cfg = raw["regimes"][regime]
    ridge_cfg = raw["evaluation"]["ridge"]

    if model_cfg.get("objective") != "causal_lm":
        raise ValueError("The paper profile requires the causal language-modeling objective")

    config = RunConfig(
        model_key=model,
        model_id=str(model_cfg["id"]),
        dataset_key=dataset,
        subjects=tuple(str(item) for item in dataset_cfg["subjects"]),
        regime=regime,
        test_fold=test_fold,
        folds=folds,
        seed=int(raw["experiment"]["seed"] if seed is None else seed),
        epochs=int(raw["experiment"]["epochs"]),
        batch_size=int(raw["experiment"]["batch_size"]),
        validation_fraction=float(raw["experiment"]["validation_fraction"]),
        context_words=int(dataset_cfg["context_words"]),
        context_groups=int(dataset_cfg["context_groups"]),
        checkpoint_eval_steps=int(dataset_cfg["checkpoint_eval_steps"]),
        noise_ceiling_threshold=float(dataset_cfg["noise_ceiling_threshold"]),
        learning_rate=float(regime_cfg["learning_rate"]),
        weight_decay=float(raw["experiment"]["weight_decay"]),
        fmri_aggregation=str(raw["experiment"]["fmri_aggregation"]),
        lm_weight=float(regime_cfg["lm_weight"]),
        brain_weight=float(regime_cfg["brain_weight"]),
        lora_rank=(None if regime_cfg["lora_rank"] is None else int(regime_cfg["lora_rank"])),
        lora_alpha=(None if regime_cfg["lora_alpha"] is None else int(regime_cfg["lora_alpha"])),
        lora_dropout=float(regime_cfg["lora_dropout"]),
        target_modules=tuple(str(item) for item in model_cfg["target_modules"]),
        ridge_lambdas=tuple(float(item) for item in ridge_cfg["lambdas"]),
        ridge_splits=int(ridge_cfg["splits"]),
        holmes_seeds=tuple(int(item) for item in raw["evaluation"]["holmes_seeds"]),
    )
    if config.folds != 4:
        raise ValueError("The paper profile requires four folds")
    if config.subjects != EXPECTED_SUBJECTS[dataset]:
        raise ValueError(f"The paper profile requires subjects {EXPECTED_SUBJECTS[dataset]}")
    if config.epochs < 1 or config.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    if not 0 < config.validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if config.context_words < 1 or config.context_groups != 5:
        raise ValueError(
            "The paper profile requires positive context_words and five context groups"
        )
    if config.checkpoint_eval_steps < 1:
        raise ValueError("checkpoint_eval_steps must be positive")
    if config.learning_rate < 0 or config.weight_decay < 0:
        raise ValueError("Learning rate and weight decay must be non-negative")
    if config.fmri_aggregation != "none":
        raise ValueError("The paper training profile preserves individual voxel targets")
    if config.lm_weight < 0 or config.brain_weight < 0:
        raise ValueError("Loss weights must be non-negative")
    if regime == "pretrained":
        if config.lora_rank is not None or config.lora_alpha is not None:
            raise ValueError("The pretrained regime must not configure LoRA")
    elif config.lora_rank is None or config.lora_alpha is None:
        raise ValueError("Fine-tuned paper regimes require LoRA rank and alpha")
    if not config.target_modules:
        raise ValueError("At least one LoRA target module is required")
    if not config.ridge_lambdas or any(value <= 0 for value in config.ridge_lambdas):
        raise ValueError("Ridge lambdas must be non-empty and positive")
    if config.ridge_splits < 2:
        raise ValueError("Ridge evaluation requires at least two splits")
    if config.holmes_seeds != (0, 1, 2, 3, 4):
        raise ValueError("The paper Holmes profile requires seeds 0, 1, 2, 3, 4")
    return config
