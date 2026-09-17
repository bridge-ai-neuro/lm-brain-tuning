from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .compat import build_hp_contexts, build_moth_contexts
from .config import EXPECTED_SUBJECTS, SUPPORTED_DATASETS, load_mapping
from .io import atomic_write_json, load_numeric_array, read_jsonl, sha256_file, write_jsonl

if TYPE_CHECKING:
    from .config import RunConfig

SCHEMA_VERSION = 1


def _referenced_files(manifest: dict[str, Any]) -> set[str]:
    files = {str(manifest["contexts"]), str(manifest["runs"])}
    for field in ("responses", "language_masks", "noise_ceilings"):
        values = manifest.get(field)
        if not isinstance(values, dict):
            raise ValueError(f"Dataset manifest field {field!r} must be a subject mapping")
        files.update(str(value) for value in values.values())
    return files


def _resolved(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _bundle_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Dataset path escapes the bundle: {value}")
    return path


def _save_array(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(array), allow_pickle=False)


@dataclass
class CanonicalDataset:
    root: Path
    manifest: dict[str, Any]
    contexts: list[str]
    context_chunks: list[list[str]]
    runs: np.ndarray
    responses: dict[str, np.ndarray]
    language_masks: dict[str, np.ndarray]
    noise_ceilings: dict[str, np.ndarray]

    @classmethod
    def load(cls, root: str | Path) -> CanonicalDataset:
        root = Path(root).resolve()
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Unsupported dataset schema: {manifest.get('schema_version')}")
        referenced = _referenced_files(manifest)
        checksums = manifest.get("checksums")
        if not isinstance(checksums, dict):
            raise ValueError("Dataset manifest must contain a checksum mapping")
        missing_checksums = sorted(referenced.difference(checksums))
        if missing_checksums:
            raise ValueError(f"Dataset manifest is missing checksums for: {missing_checksums}")
        for relative, expected in checksums.items():
            path = _bundle_path(root, relative)
            if not path.is_file() or sha256_file(path) != expected:
                raise ValueError(f"Dataset checksum mismatch: {relative}")
        rows = read_jsonl(_bundle_path(root, manifest["contexts"]))
        dataset = cls(
            root=root,
            manifest=manifest,
            contexts=[str(row["text"]) for row in rows],
            context_chunks=[[str(item) for item in row["chunks"]] for row in rows],
            runs=load_numeric_array(_bundle_path(root, manifest["runs"])),
            responses={
                subject: load_numeric_array(_bundle_path(root, relative))
                for subject, relative in manifest["responses"].items()
            },
            language_masks={
                subject: load_numeric_array(_bundle_path(root, relative)).astype(bool)
                for subject, relative in manifest["language_masks"].items()
            },
            noise_ceilings={
                subject: load_numeric_array(_bundle_path(root, relative))
                for subject, relative in manifest["noise_ceilings"].items()
            },
        )
        dataset.validate()
        return dataset

    @property
    def dataset_key(self) -> str:
        return str(self.manifest["dataset"])

    @property
    def subjects(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.manifest["subjects"])

    def combined_masks(self, threshold: float) -> dict[str, np.ndarray]:
        return {
            subject: np.logical_and(
                self.language_masks[subject], self.noise_ceilings[subject] > threshold
            )
            for subject in self.subjects
        }

    def validate(self) -> None:
        if self.dataset_key not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported dataset in manifest: {self.dataset_key}")
        if int(self.manifest.get("folds", -1)) != 4:
            raise ValueError("The paper profile requires exactly four folds")
        if self.subjects != EXPECTED_SUBJECTS[self.dataset_key]:
            raise ValueError(
                f"Subject order must be {EXPECTED_SUBJECTS[self.dataset_key]}, got {self.subjects}"
            )
        expected_subjects = set(self.subjects)
        for field in ("responses", "language_masks", "noise_ceilings"):
            actual_subjects = set(self.manifest[field])
            if actual_subjects != expected_subjects:
                raise ValueError(
                    f"Manifest {field} subjects must match the declared subjects exactly"
                )
        samples = len(self.contexts)
        if samples == 0:
            raise ValueError("Dataset must contain at least one sample")
        if len(self.context_chunks) != samples or len(self.runs) != samples:
            raise ValueError("Contexts, chunks, and run mapping must have equal lengths")
        if self.runs.ndim != 1 or not np.issubdtype(self.runs.dtype, np.integer):
            raise ValueError("Run mapping must be a one-dimensional integer array")
        if set(np.unique(self.runs).tolist()) != {0, 1, 2, 3}:
            raise ValueError("The paper profile requires run/fold identifiers 0, 1, 2, 3")
        expected_groups = int(self.manifest["context_groups"])
        if expected_groups != 5:
            raise ValueError("The paper profile requires exactly five context groups")
        if int(self.manifest["context_words"]) < 1:
            raise ValueError("context_words must be positive")
        if any(len(chunks) != expected_groups for chunks in self.context_chunks):
            raise ValueError(f"Every context must contain exactly {expected_groups} groups")
        for subject in self.subjects:
            response = self.responses[subject]
            if response.ndim != 2 or response.shape[0] != samples:
                raise ValueError(f"Invalid response shape for {subject}: {response.shape}")
            voxel_count = response.shape[1]
            if self.language_masks[subject].shape != (voxel_count,):
                raise ValueError(f"Invalid language mask shape for {subject}")
            if self.noise_ceilings[subject].shape != (voxel_count,):
                raise ValueError(f"Invalid noise ceiling shape for {subject}")

    def validate_for_run(self, config: RunConfig, subject: str) -> None:
        """Reject incompatible inputs before any model or output is created."""
        if self.dataset_key != config.dataset_key:
            raise ValueError("Dataset and run configuration do not match")
        if self.subjects != config.subjects:
            raise ValueError("Dataset and run configuration subject order do not match")
        if int(self.manifest["folds"]) != config.folds:
            raise ValueError("Dataset and run configuration fold counts do not match")
        if int(self.manifest["context_words"]) != config.context_words:
            raise ValueError("Dataset and run configuration context_words do not match")
        if int(self.manifest["context_groups"]) != config.context_groups:
            raise ValueError("Dataset and run configuration context_groups do not match")
        if subject not in self.subjects:
            raise ValueError(f"Unknown subject: {subject}")
        selected_voxels = self.combined_masks(config.noise_ceiling_threshold)[subject]
        if not np.any(selected_voxels):
            raise ValueError(
                f"Subject {subject} has no language-ROI voxels above the noise-ceiling threshold"
            )


def _load_contexts(
    source: dict[str, Any], base: Path, dataset: str, context_words: int, context_groups: int
) -> tuple[list[str], list[list[str]]]:
    if "contexts" in source:
        rows = read_jsonl(_resolved(base, source["contexts"]))
        return (
            [str(row["text"]) for row in rows],
            [[str(value) for value in row["chunks"]] for row in rows],
        )

    stimulus = source["stimulus"]
    allow_pickle = bool(stimulus.get("trusted_object_arrays", False))
    words = np.load(_resolved(base, stimulus["words"]), allow_pickle=allow_pickle).tolist()
    word_times = load_numeric_array(_resolved(base, stimulus["word_times"]))
    fmri_times = load_numeric_array(_resolved(base, stimulus["fmri_times"]))
    if dataset == "hp":
        contexts, chunks = build_hp_contexts(
            words, word_times, fmri_times, context_words=context_words
        )
        for item in chunks:
            while len(item) < context_groups:
                item.insert(0, "")
        return contexts, chunks
    return build_moth_contexts(
        words,
        word_times,
        fmri_times,
        context_words=context_words,
        context_groups=context_groups,
    )


def prepare_dataset(source_manifest: str | Path, output: str | Path) -> Path:
    """Convert a source descriptor into the safe canonical on-disk contract."""
    source_path = Path(source_manifest).resolve()
    source = load_mapping(source_path)
    source_base = source_path.parent
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty dataset directory: {output}")

    dataset = str(source["dataset"])
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported dataset: {dataset}")
    subjects = tuple(str(value) for value in source["subjects"])
    if subjects != EXPECTED_SUBJECTS[dataset]:
        raise ValueError(f"Expected subjects {EXPECTED_SUBJECTS[dataset]}, got {subjects}")
    context_words = int(source.get("context_words", 100 if dataset == "moth" else 20))
    context_groups = int(source.get("context_groups", 5))
    if context_words < 1 or context_groups != 5:
        raise ValueError(
            "The paper profile requires positive context_words and five context groups"
        )
    contexts, chunks = _load_contexts(source, source_base, dataset, context_words, context_groups)
    sample_count = len(contexts)
    if sample_count == 0:
        raise ValueError("Source dataset must contain at least one sample")
    if sample_count != len(chunks):
        raise ValueError("Contexts and context chunks must have equal lengths")
    if any(len(group) != context_groups for group in chunks):
        raise ValueError(f"Every source context must contain exactly {context_groups} groups")
    raw_runs = load_numeric_array(_resolved(source_base, source["runs"]))
    if raw_runs.ndim != 1 or not np.issubdtype(raw_runs.dtype, np.integer):
        raise ValueError("Source run mapping must be a one-dimensional integer array")
    runs = raw_runs.astype(np.int64, copy=False)
    if len(runs) != sample_count:
        raise ValueError("Source contexts and run mapping must have equal lengths")
    if set(np.unique(runs).tolist()) != {0, 1, 2, 3}:
        raise ValueError("Source run mapping must contain fold identifiers 0, 1, 2, 3")

    expected_subjects = set(subjects)
    for field in ("responses", "language_masks", "noise_ceilings"):
        values = source.get(field)
        if not isinstance(values, dict) or set(values) != expected_subjects:
            raise ValueError(f"Source {field} must map every paper subject exactly once")

    for subject in subjects:
        response = load_numeric_array(_resolved(source_base, source["responses"][subject]))
        language_mask = load_numeric_array(
            _resolved(source_base, source["language_masks"][subject])
        )
        noise_ceiling = load_numeric_array(
            _resolved(source_base, source["noise_ceilings"][subject])
        )
        if response.ndim != 2 or response.shape[0] != sample_count:
            raise ValueError(f"Invalid source response shape for {subject}: {response.shape}")
        voxel_count = response.shape[1]
        if language_mask.shape != (voxel_count,):
            raise ValueError(f"Invalid source language mask shape for {subject}")
        if noise_ceiling.shape != (voxel_count,):
            raise ValueError(f"Invalid source noise ceiling shape for {subject}")

    output.mkdir(parents=True, exist_ok=True)
    contexts_path = output / "contexts.jsonl"
    write_jsonl(
        contexts_path,
        ({"text": text, "chunks": group} for text, group in zip(contexts, chunks, strict=False)),
    )
    runs_path = output / "runs.npy"
    _save_array(runs_path, runs)

    files: dict[str, dict[str, str]] = {
        "responses": {},
        "language_masks": {},
        "noise_ceilings": {},
    }
    for subject in subjects:
        source_response = _resolved(source_base, source["responses"][subject])
        source_mask = _resolved(source_base, source["language_masks"][subject])
        source_noise = _resolved(source_base, source["noise_ceilings"][subject])
        destinations = {
            "responses": output / "responses" / f"sub-{subject}.npy",
            "language_masks": output / "language_masks" / f"sub-{subject}.npy",
            "noise_ceilings": output / "noise_ceilings" / f"sub-{subject}.npy",
        }
        _save_array(destinations["responses"], load_numeric_array(source_response))
        _save_array(destinations["language_masks"], load_numeric_array(source_mask).astype(bool))
        _save_array(destinations["noise_ceilings"], load_numeric_array(source_noise))
        for kind, destination in destinations.items():
            files[kind][subject] = destination.relative_to(output).as_posix()

    checksums = {
        path.relative_to(output).as_posix(): sha256_file(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset,
        "subjects": list(subjects),
        "folds": 4,
        "context_words": context_words,
        "context_groups": context_groups,
        "contexts": contexts_path.relative_to(output).as_posix(),
        "runs": runs_path.relative_to(output).as_posix(),
        **files,
        "source": {
            "descriptor": source_path.name,
            "citation": source.get("citation"),
            "url": source.get("url"),
        },
        "checksums": checksums,
    }
    atomic_write_json(output / "manifest.json", manifest)
    CanonicalDataset.load(output)
    return output / "manifest.json"
