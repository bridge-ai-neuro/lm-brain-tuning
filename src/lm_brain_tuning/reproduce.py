from __future__ import annotations

import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import SUPPORTED_DATASETS, SUPPORTED_MODELS, SUPPORTED_REGIMES, load_mapping
from .io import atomic_write_json


@dataclass(frozen=True)
class ReproductionJob:
    model: str
    dataset: str
    regime: str
    fold: int
    subject: str

    @property
    def identifier(self) -> str:
        return f"{self.model}-{self.dataset}-{self.regime}-fold{self.fold}-sub{self.subject}"


def build_matrix(config_path: str | Path) -> list[ReproductionJob]:
    raw = load_mapping(config_path)
    folds = int(raw["experiment"]["folds"])
    jobs: list[ReproductionJob] = []
    for model in SUPPORTED_MODELS:
        for dataset in SUPPORTED_DATASETS:
            subjects = [str(value) for value in raw["datasets"][dataset]["subjects"]]
            for regime in SUPPORTED_REGIMES:
                for fold in range(folds):
                    for subject in subjects:
                        jobs.append(ReproductionJob(model, dataset, regime, fold, subject))
    return jobs


def commands_for_job(
    job: ReproductionJob,
    *,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
) -> list[list[str]]:
    raw = load_mapping(config_path)
    base = [
        "lm-brain-tuning",
        "--config",
        str(Path(config_path).resolve()),
    ]
    common = [
        "--model",
        job.model,
        "--dataset",
        job.dataset,
        "--regime",
        job.regime,
        "--fold",
        str(job.fold),
        "--subject",
        job.subject,
    ]
    data = Path(data_root).resolve() / job.dataset
    output = Path(output_root).resolve()
    commands: list[list[str]] = []
    if job.regime != "pretrained":
        validation_root = output / "selection-training"
        evaluation_steps = int(raw["datasets"][job.dataset]["checkpoint_eval_steps"])
        commands.append(
            base
            + [
                "train",
                *common,
                "--data",
                str(data),
                "--output",
                str(validation_root),
                "--evaluation-steps",
                str(evaluation_steps),
            ]
        )
        validation_output = (
            validation_root
            / f"{job.model}-{job.dataset}-{job.regime}-fold{job.fold}"
            / f"sub-{job.subject}"
        )
        validation_selection = validation_output / "selection.json"
        commands.append(
            base
            + [
                "select-checkpoint",
                "--log",
                str(validation_output / "log.csv"),
                "--checkpoint-root",
                str(validation_output),
                "--output",
                str(validation_selection),
            ]
        )
        commands.append(
            base
            + [
                "train",
                *common,
                "--data",
                str(data),
                "--output",
                str(output / "training"),
                "--selection",
                str(validation_selection),
            ]
        )
        final_output = (
            output
            / "training"
            / f"{job.model}-{job.dataset}-{job.regime}-fold{job.fold}"
            / f"sub-{job.subject}"
        )
        selection = final_output / "selection.json"
    else:
        model_path = str(raw["models"][job.model]["id"])
    evaluation = base + [
        "evaluate-brain",
        *common,
        "--data",
        str(data),
    ]
    if job.regime == "pretrained":
        evaluation.extend(["--model-path", model_path])
    else:
        evaluation.extend(["--selection", str(selection)])
    evaluation.extend(["--output", str(output / "brain-evaluation")])
    commands.append(evaluation)
    return commands


def materialize_reproduction(
    *,
    config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
) -> tuple[Path, list[list[str]]]:
    output = Path(output_root).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty reproduction output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    jobs = build_matrix(config_path)
    job_dir = output / "jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []
    for job in jobs:
        stages = commands_for_job(
            job, config_path=config_path, data_root=data_root, output_root=output
        )
        script = job_dir / f"{job.identifier}.sh"
        script.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + "\n".join(shlex.join(stage) for stage in stages)
            + "\n",
            encoding="utf-8",
        )
        script.chmod(0o755)
        commands.append(["bash", str(script)])
    atomic_write_json(
        output / "matrix.json",
        {"jobs": [asdict(job) | {"identifier": job.identifier} for job in jobs]},
    )
    command_path = output / "commands.txt"
    command_path.write_text(
        "\n".join(shlex.join(command) for command in commands) + "\n", encoding="utf-8"
    )
    return command_path, commands


def run_local(commands: list[list[str]]) -> None:
    for command in commands:
        subprocess.run(command, check=True)


def write_slurm_launcher(command_path: str | Path, output: str | Path) -> Path:
    command_path = Path(command_path).resolve()
    line_count = sum(1 for line in command_path.read_text(encoding="utf-8").splitlines() if line)
    if line_count < 1:
        raise ValueError("Cannot create a Slurm launcher for an empty command file")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Slurm launcher: {output}")
    script = f"""#!/usr/bin/env bash
#SBATCH --job-name=lm-brain-tuning
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --array=1-{line_count}
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
set -euo pipefail
command=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {shlex.quote(str(command_path))})
test -n "$command"
bash -lc "$command"
"""
    output.write_text(script, encoding="utf-8")
    output.chmod(0o755)
    return output
