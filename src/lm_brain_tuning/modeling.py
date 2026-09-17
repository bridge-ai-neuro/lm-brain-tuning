from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import AutoModelForCausalLM, PretrainedConfig, PreTrainedModel

from .compat import concatenate_average_tr, pearson_loss


class BrainTuningConfig(PretrainedConfig):
    model_type = "lm_brain_tuning"

    def __init__(
        self,
        backbone_name: str = "gpt2",
        num_regression_labels: int = 1,
        context_groups: int = 5,
        pad_token_id: int = -1,
        lm_weight: float = 1.0,
        brain_weight: float = 0.0,
        lora_rank: int | None = None,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        target_modules: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.num_regression_labels = num_regression_labels
        self.context_groups = context_groups
        self.regression_pad_token_id = pad_token_id
        self.lm_weight = lm_weight
        self.brain_weight = brain_weight
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules


class BrainTuningModel(PreTrainedModel):
    """Causal-LM backbone plus the paper's voxel regression head."""

    config_class = BrainTuningConfig
    base_model_prefix = "backbone_lm"

    def __init__(self, config: BrainTuningConfig) -> None:
        super().__init__(config)
        self.backbone_lm = AutoModelForCausalLM.from_pretrained(config.backbone_name)
        hidden_size = int(self.backbone_lm.config.hidden_size)
        self.regression_head = nn.Linear(
            hidden_size * config.context_groups, config.num_regression_labels
        )
        self.register_buffer("weight_lm", torch.tensor(config.lm_weight, dtype=torch.float64))
        self.register_buffer(
            "weight_regression", torch.tensor(config.brain_weight, dtype=torch.float64)
        )
        # Keep regression-head initialization before LoRA initialization.
        if config.lora_rank is not None:
            self.backbone_lm = _wrap_lora(
                self.backbone_lm,
                rank=config.lora_rank,
                alpha=config.lora_alpha,
                dropout=config.lora_dropout,
                target_modules=tuple(config.target_modules),
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        labels_regression: torch.Tensor | None = None,
        mapping: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        output = self.backbone_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )
        if mapping is None:
            raise ValueError("mapping is required by the paper representation")
        regression_input = concatenate_average_tr(output.hidden_states[-1], mapping)
        regression_output = self.regression_head(regression_input)
        regression_loss = torch.zeros((), device=input_ids.device)
        if labels_regression is not None:
            if labels_regression.ndim == 1:
                labels_regression = labels_regression.unsqueeze(1)
            losses = [
                pearson_loss(labels_regression[:, index], regression_output[:, index].flatten())
                for index in range(self.config.num_regression_labels)
            ]
            regression_loss = torch.stack(losses).mean()

        loss = None
        if labels is not None and labels_regression is not None:
            loss = self.weight_lm * output.loss + self.weight_regression * regression_loss
        return {
            "loss": loss,
            "logits": output.logits,
            "regression": regression_output,
            "lm_loss": output.loss,
            "regression_loss": regression_loss,
            "additional_metrics": {
                "lm_loss": output.loss,
                "regression_loss": regression_loss,
            },
        }

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        # A single PyTorch state file avoids PEFT/shared-tensor aliasing.
        kwargs.setdefault("safe_serialization", False)
        kwargs.setdefault("max_shard_size", "50GB")
        super().save_pretrained(save_directory, **kwargs)

    def merge_lora(self) -> BrainTuningModel:
        """Merge adapters into the base language model."""
        if hasattr(self.backbone_lm, "merge_and_unload"):
            self.backbone_lm = self.backbone_lm.merge_and_unload()
            self.config.lora_rank = None
            self.config.lora_alpha = None
            self.config.target_modules = None
        return self

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs: Any):
        """Load without the outer Transformer's meta-device constructor path."""
        source = Path(pretrained_model_name_or_path)
        config = kwargs.pop("config", None) or BrainTuningConfig.from_pretrained(source)
        model = cls(config)
        state_dict = kwargs.pop("state_dict", None)
        if state_dict is None:
            weights = source / "pytorch_model.bin"
            safe_weights = source / "model.safetensors"
            if weights.exists():
                state_dict = torch.load(weights, map_location="cpu", weights_only=True)
            elif safe_weights.exists():
                from safetensors.torch import load_file

                state_dict = load_file(safe_weights)
            else:
                raise FileNotFoundError(f"Checkpoint weights not found in {source}")
        incompatible = model.load_state_dict(state_dict, strict=False)
        # SafeTensors deliberately stores only one side of tied parameters.  The
        # missing aliases below are recreated by ``tie_weights`` and therefore
        # do not alter a checkpoint's values.  Every other mismatch remains a
        # hard error so a partial or incompatible checkpoint cannot be used.
        tied_weight_suffixes = (
            "cls.predictions.decoder.weight",
            "cls.predictions.decoder.bias",
            "lm_head.weight",
        )
        missing = [
            key for key in incompatible.missing_keys if not key.endswith(tied_weight_suffixes)
        ]
        unexpected = list(incompatible.unexpected_keys)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint state mismatch: "
                f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
            )
        model.backbone_lm.tie_weights()
        return model


def _wrap_lora(
    backbone: Any,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
) -> Any:
    from peft import LoraConfig, TaskType, get_peft_model

    lora = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        init_lora_weights=True,
    )
    return get_peft_model(backbone, lora)


def apply_paper_lora(
    model: BrainTuningModel,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
) -> BrainTuningModel:
    model.config.lora_rank = rank
    model.config.lora_alpha = alpha
    model.config.lora_dropout = dropout
    model.config.target_modules = list(target_modules)
    model.backbone_lm = _wrap_lora(
        model.backbone_lm,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    return model
