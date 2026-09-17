from __future__ import annotations

import re
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


def clean_stimulus(text: str, *, remove_format_chars: bool = True) -> str:
    text = re.sub(r"\s*—\s*", "—", re.sub(r"--", "—", text))
    while re.search(r"\.\s+\.", text):
        text = re.sub(r"\.\s+\.", "..", text)
    if remove_format_chars:
        text = re.sub(r"@|\+", "", text)
    return text


def build_hp_contexts(
    words: Sequence[str],
    word_times: Sequence[float],
    fmri_times: Sequence[float],
    *,
    context_words: int = 20,
) -> tuple[list[str], list[list[str]]]:
    """Build Harry Potter contexts and four-word groups."""
    contexts: list[str] = []
    chunks_by_context: list[list[str]] = []
    word_time_pairs = list(zip(word_times, words, strict=False))
    for fmri_time in fmri_times:
        selected = [word for time, word in word_time_pairs if time <= fmri_time][-context_words:]
        chunks = [" ".join(selected[index : index + 4]) for index in range(0, len(selected), 4)]
        for index, chunk in enumerate(chunks):
            cleaned = clean_stimulus(chunk)
            if index != 0:
                chunks[index] = " " + cleaned
        contexts.append(clean_stimulus(" ".join(selected)))
        chunks_by_context.append(chunks)
    return contexts, chunks_by_context


def stitch_by_previous_last(arrays: Iterable[Sequence[float]]) -> np.ndarray:
    arrays = [np.asarray(array, dtype=float) for array in arrays]
    parts: list[np.ndarray] = []
    last_global = 0.0
    for index, array in enumerate(arrays):
        if array.size == 0:
            continue
        shifted = array.copy() if index == 0 else array + last_global
        parts.append(shifted)
        last_global = float(shifted[-1])
    return np.hstack(parts) if parts else np.array([], dtype=float)


def build_moth_contexts(
    words: Sequence[str],
    word_times: Sequence[float],
    fmri_times: Sequence[float],
    *,
    context_words: int = 20,
    context_groups: int = 5,
) -> tuple[list[str], list[list[str]]]:
    """Build ordered Subset Moth contexts and TR-aligned groups."""
    word_to_tr = [bisect_right(fmri_times, time) for time in word_times]
    triples = list(zip(word_times, words, word_to_tr, strict=False))
    contexts: list[str] = []
    chunks_by_context: list[list[str]] = []
    for fmri_time in fmri_times:
        selected = [(word, tr) for time, word, tr in triples if time <= fmri_time][-context_words:]
        grouped: dict[int, list[str]] = defaultdict(list)
        for word, tr in selected:
            grouped[tr].append(str(word))
        chunks = [" ".join(group) for group in grouped.values()][-context_groups:]
        chunks = [chunk if index == 0 else " " + chunk for index, chunk in enumerate(chunks)]
        while len(chunks) < context_groups:
            chunks.insert(0, "")
        contexts.append(" ".join(word for word, _ in selected))
        chunks_by_context.append(chunks)
    return contexts, chunks_by_context


def select_runs(
    selected_runs: Sequence[int],
    contexts: Sequence[str],
    context_chunks: Sequence[Sequence[str]],
    responses: dict[str, np.ndarray],
    runs: Sequence[int],
) -> tuple[list[str], list[list[str]], dict[str, np.ndarray]]:
    """Select runs while retaining the input sample order."""
    selected = set(selected_runs)
    selected_contexts = [
        value for value, run in zip(contexts, runs, strict=False) if run in selected
    ]
    selected_chunks = [
        list(value) for value, run in zip(context_chunks, runs, strict=False) if run in selected
    ]
    selected_responses = {
        subject: np.asarray(
            [value for value, run in zip(array, runs, strict=False) if run in selected]
        )
        for subject, array in responses.items()
    }
    return selected_contexts, selected_chunks, selected_responses


def normalize_per_runs(
    responses: dict[str, np.ndarray], runs: Sequence[int], selected_runs: Sequence[int]
) -> dict[str, np.ndarray]:
    from scipy.stats import zscore

    runs = np.asarray(runs)
    filtered_runs = runs[np.isin(runs, selected_runs)]
    normalized: dict[str, np.ndarray] = {}
    for subject, subject_data in responses.items():
        data = np.nan_to_num(subject_data)
        output = np.zeros_like(data)
        for run in selected_runs:
            mask = filtered_runs == run
            output[mask] = zscore(data[mask])
        normalized[subject] = np.nan_to_num(output)
    return normalized


def normalize_all(responses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    from scipy.stats import zscore

    return {
        subject: np.nan_to_num(zscore(np.nan_to_num(values)))
        for subject, values in responses.items()
    }


def filter_voxels(
    responses: dict[str, np.ndarray], masks: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {
        subject: responses[subject][:, np.asarray(masks[subject], dtype=bool)]
        for subject in responses
    }


def aggregate_voxels(responses: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {subject: np.mean(values, axis=1) for subject, values in responses.items()}


def correlation_by_target(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    from scipy.stats import zscore

    return np.mean(zscore(prediction) * zscore(target), axis=0)


def tokenize_and_aggregate(
    model_name: str,
    tokenizer: Any,
    contexts: Sequence[str],
    context_chunks: Sequence[Sequence[str]],
    extraction_method: str = "concatenate_avg_tr",
) -> tuple[dict[str, Any], Any]:
    """Tokenize contexts and build flatten-then-pad group mappings."""
    import torch

    if "bert" not in model_name:
        tokenizer.pad_token = tokenizer.eos_token
    if extraction_method != "concatenate_avg_tr":
        encoded = tokenizer(list(contexts), return_tensors="pt", padding=True)
        mapping = torch.full((encoded["input_ids"].size(0),), -1)
        return dict(encoded), mapping

    sequences: list[Any] = []
    mappings: list[Any] = []
    for chunks in context_chunks:
        result = tokenizer(list(chunks), return_tensors="pt", padding=True)
        input_ids = result["input_ids"]
        flattened = input_ids[input_ids != tokenizer.pad_token_id]
        sequences.append(flattened)
        mapping_values: list[int] = []
        for group, row in enumerate(input_ids):
            mapping_values.extend([group] * int((row != tokenizer.pad_token_id).sum().item()))
        mappings.append(torch.tensor(mapping_values, dtype=torch.long))

    maximum = max(len(sequence) for sequence in sequences)
    padded_ids: list[Any] = []
    attention_masks: list[Any] = []
    padded_mappings: list[Any] = []
    for sequence, mapping in zip(sequences, mappings, strict=False):
        padding = maximum - len(sequence)
        padded_ids.append(
            torch.cat([sequence, torch.full((padding,), tokenizer.pad_token_id, dtype=torch.long)])
        )
        attention_masks.append(
            torch.cat(
                [
                    torch.ones(len(sequence), dtype=torch.long),
                    torch.zeros(padding, dtype=torch.long),
                ]
            )
        )
        padded_mappings.append(torch.cat([mapping, torch.full((padding,), -1, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(padded_ids),
        "attention_mask": torch.stack(attention_masks),
    }, torch.stack(padded_mappings)


def concatenate_average_tr(last_hidden_state: Any, mapping: Any) -> Any:
    """Apply five-group mean pooling for training and evaluation."""
    import torch

    batch_size, _, feature_dim = last_hidden_state.shape
    max_group = int(mapping[mapping != -1].max().item()) + 1
    result = torch.full(
        (batch_size, feature_dim * max_group),
        -1.0,
        device=last_hidden_state.device,
        dtype=last_hidden_state.dtype,
    )
    mapping = mapping.to(last_hidden_state.device)
    for item in range(batch_size):
        valid = mapping[item] != -1
        vectors = last_hidden_state[item][valid]
        groups = mapping[item][valid]
        averages = []
        for group in range(max_group):
            group_mask = groups == group
            averages.append(
                vectors[group_mask].mean(dim=0)
                if group_mask.any()
                else torch.zeros(feature_dim, device=vectors.device, dtype=vectors.dtype)
            )
        result[item] = torch.cat(averages)
    return result


def pearson_loss(y_true: Any, y_pred: Any) -> Any:
    """Compute the global Pearson loss used for brain tuning."""
    import torch

    true_centered = y_true - torch.mean(y_true)
    pred_centered = y_pred - torch.mean(y_pred)
    numerator = torch.sum(true_centered * pred_centered)
    denominator = torch.sqrt(torch.sum(true_centered**2)) * torch.sqrt(torch.sum(pred_centered**2))
    return -(numerator / denominator)
