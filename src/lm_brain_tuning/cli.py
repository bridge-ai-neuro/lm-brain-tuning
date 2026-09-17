from __future__ import annotations

import argparse
import json
import subprocess
from importlib.resources import files
from pathlib import Path

from .config import SUPPORTED_DATASETS, SUPPORTED_MODELS, SUPPORTED_REGIMES, resolve_run_config


def _add_run_arguments(parser: argparse.ArgumentParser, *, subject: bool = True) -> None:
    parser.add_argument("--model", choices=SUPPORTED_MODELS, required=True)
    parser.add_argument("--dataset", choices=SUPPORTED_DATASETS, required=True)
    parser.add_argument("--regime", choices=SUPPORTED_REGIMES, required=True)
    parser.add_argument("--fold", type=int, required=True)
    if subject:
        parser.add_argument("--subject", required=True)
    parser.add_argument("--seed", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lm-brain-tuning")
    parser.add_argument(
        "--config", default=str(files("lm_brain_tuning").joinpath("paper.yaml"))
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--source-manifest", required=True)
    prepare.add_argument("--output", required=True)

    train_parser = subparsers.add_parser("train")
    _add_run_arguments(train_parser)
    train_parser.add_argument("--data", required=True)
    train_parser.add_argument("--output", required=True)
    selected = train_parser.add_mutually_exclusive_group()
    selected.add_argument("--selected-step", type=int)
    selected.add_argument("--selection")
    train_parser.add_argument("--evaluation-steps", type=int)

    checkpoint = subparsers.add_parser("select-checkpoint")
    checkpoint.add_argument("--log", required=True)
    checkpoint.add_argument("--output", required=True)
    checkpoint.add_argument("--checkpoint-root")

    brain = subparsers.add_parser("evaluate-brain")
    _add_run_arguments(brain)
    brain.add_argument("--data", required=True)
    brain_model = brain.add_mutually_exclusive_group(required=True)
    brain_model.add_argument("--model-path")
    brain_model.add_argument("--selection")
    brain.add_argument("--output", required=True)
    brain.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    brain.add_argument("--feature-batch-size", type=int, default=8)

    holmes = subparsers.add_parser("evaluate-holmes")
    holmes_mode = holmes.add_mutually_exclusive_group(required=True)
    holmes_mode.add_argument("--run", action="store_true")
    holmes_mode.add_argument("--normalize-csv")
    holmes.add_argument("--checkout")
    holmes.add_argument("--model-path")
    holmes.add_argument("--output", required=True)
    holmes.add_argument("--entrypoint", default="src/investigate.py")
    holmes.add_argument("--version", default="flash-holmes")
    holmes.add_argument("--system")
    holmes.add_argument("--comparison-unit")
    holmes.add_argument("--analysis-unit")
    holmes.add_argument("--aggregation-unit")
    holmes.add_argument("--task-metadata")

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--holmes-csv", action="append", required=True)
    analyze.add_argument("--baseline", required=True)
    analyze.add_argument("--output", required=True)

    reproduce = subparsers.add_parser("reproduce")
    reproduce.add_argument("--data-root", required=True)
    reproduce.add_argument("--output", required=True)
    mode = reproduce.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--local", action="store_true")
    mode.add_argument("--slurm", action="store_true")
    reproduce.add_argument("--submit", action="store_true")
    return parser


def _resolve(args: argparse.Namespace):
    return resolve_run_config(
        args.config,
        model=args.model,
        dataset=args.dataset,
        regime=args.regime,
        test_fold=args.fold,
        seed=args.seed,
    )


def _load_selection(path: str) -> dict[str, object]:
    selection = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(selection, dict):
        raise SystemExit("Selection manifest must contain a JSON object")
    return selection


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare-data":
        from .data import prepare_dataset

        print(prepare_dataset(args.source_manifest, args.output))
    elif args.command == "train":
        from .training import train

        config = _resolve(args)
        selected_step = args.selected_step
        selection_metadata = None
        if args.selection:
            selection_metadata = _load_selection(args.selection)
            selected_step = int(selection_metadata["step"])
        print(
            train(
                config,
                dataset_root=args.data,
                subject=args.subject,
                output_root=args.output,
                selected_step=selected_step,
                selection_metadata=selection_metadata,
                evaluation_steps=args.evaluation_steps,
            )
        )
    elif args.command == "select-checkpoint":
        from .checkpoints import select_checkpoint

        print(json.dumps(select_checkpoint(args.log, args.output, args.checkpoint_root), indent=2))
    elif args.command == "evaluate-brain":
        from .evaluation import evaluate_brain

        config = _resolve(args)
        model_path = args.model_path
        selection_metadata = None
        if args.selection:
            selection_metadata = _load_selection(args.selection)
            model_path = selection_metadata.get("checkpoint_path")
            if not model_path:
                raise SystemExit("Selection manifest does not contain checkpoint_path")
        print(
            evaluate_brain(
                config,
                dataset_root=args.data,
                subject=args.subject,
                model_path=str(model_path),
                output_root=args.output,
                device=args.device,
                feature_batch_size=args.feature_batch_size,
                selection_metadata=selection_metadata,
            )
        )
    elif args.command == "evaluate-holmes":
        from .holmes import normalize_holmes_csv, run_holmes

        if args.normalize_csv:
            required = {
                "--system": args.system,
                "--comparison-unit": args.comparison_unit,
                "--analysis-unit": args.analysis_unit,
                "--aggregation-unit": args.aggregation_unit,
            }
            missing = [flag for flag, value in required.items() if not value]
            if missing:
                raise SystemExit(f"{', '.join(missing)} required with --normalize-csv")
            print(
                normalize_holmes_csv(
                    args.normalize_csv,
                    args.output,
                    args.system,
                    args.comparison_unit,
                    args.analysis_unit,
                    args.aggregation_unit,
                    args.task_metadata,
                )
            )
        else:
            if not args.checkout or not args.model_path:
                raise SystemExit("--checkout and --model-path are required with --run")
            print(
                run_holmes(
                    checkout=args.checkout,
                    model_path=args.model_path,
                    output_dir=args.output,
                    entrypoint=args.entrypoint,
                    version=args.version,
                )
            )
    elif args.command == "analyze":
        from .analysis import analyze_holmes

        print(analyze_holmes(args.holmes_csv, baseline=args.baseline, output_dir=args.output))
    elif args.command == "reproduce":
        from .reproduce import materialize_reproduction, run_local, write_slurm_launcher

        if args.submit and not args.slurm:
            raise SystemExit("--submit is only valid together with --slurm")
        command_path, commands = materialize_reproduction(
            config_path=args.config,
            data_root=args.data_root,
            output_root=args.output,
        )
        if args.dry_run:
            print(command_path)
        elif args.local:
            run_local(commands)
        else:
            launcher = write_slurm_launcher(command_path, Path(args.output) / "reproduce.sbatch")
            print(launcher)
            if args.submit:
                subprocess.run(["sbatch", str(launcher)], check=True)
    return 0
