"""Command-line entry point: builds ExperimentConfig from defaults + flags."""

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

from brain_decoder.config import (
    AnalysisConfig,
    EmbeddingsConfig,
    ExperimentConfig,
    ExperimentPaths,
    ModelConfig,
    PipelineStages,
    TrainConfig,
    experiment_config,
    nsd_paths,
)
from brain_decoder.run import compute_embeddings, run_pipeline
from brain_decoder.utils.device import default_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Brain decoder: run embeddings, training, and/or analysis.",
    )

    stages = parser.add_argument_group("stages", "Choose --all or any combination.")
    stages.add_argument("--all", action="store_true", help="embeddings + train + analyze")
    stages.add_argument("--embeddings", action="store_true")
    stages.add_argument("--train", action="store_true")
    stages.add_argument("--analyze", action="store_true")
    stages.add_argument("--plot", action="store_true", help="heatmap from analysis cache")

    data = parser.add_argument_group("data")
    data.add_argument("--data-root", type=Path, default=argparse.SUPPRESS)
    data.add_argument("--subject", default=argparse.SUPPRESS)
    data.add_argument("--hemisphere", choices=("lh", "rh"), default=argparse.SUPPRESS)
    data.add_argument("--fmri-path", type=Path, default=argparse.SUPPRESS)
    data.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS)
    data.add_argument("--analysis-cache", type=Path, default=argparse.SUPPRESS)
    data.add_argument("--heatmap", type=Path, default=argparse.SUPPRESS)

    train = parser.add_argument_group("train")
    train.add_argument("--no-ridge", action="store_true", default=argparse.SUPPRESS)
    train.add_argument("--epochs", type=int, default=argparse.SUPPRESS)
    train.add_argument("--lr", type=float, default=argparse.SUPPRESS)
    train.add_argument("--weight-decay", type=float, default=argparse.SUPPRESS)
    train.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    train.add_argument("--val-size", type=int, default=argparse.SUPPRESS)
    train.add_argument("--eval-every", type=int, default=argparse.SUPPRESS)
    train.add_argument("--tau", type=float, default=argparse.SUPPRESS)
    train.add_argument("--grad-clip", type=float, default=argparse.SUPPRESS)
    train.add_argument("--num-workers", type=int, default=argparse.SUPPRESS)

    model = parser.add_argument_group("model")
    model.add_argument("--embed-dim", type=int, default=argparse.SUPPRESS)
    model.add_argument("--hidden-dim", type=int, default=argparse.SUPPRESS)
    model.add_argument("--dropout", type=float, default=argparse.SUPPRESS)

    emb = parser.add_argument_group("embeddings")
    emb.add_argument("--clip-model", default=argparse.SUPPRESS)

    analysis = parser.add_argument_group("analysis")
    analysis.add_argument("--rsa-bootstrap", type=int, default=argparse.SUPPRESS)
    analysis.add_argument("--pca-components", type=int, default=argparse.SUPPRESS)

    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=argparse.SUPPRESS,
        help="force device (default: cuda if available)",
    )

    return parser


def _section_overrides(section_cls, args: argparse.Namespace, mapping: dict[str, str]) -> dict | None:
    overrides = {}
    for field_name, arg_name in mapping.items():
        if hasattr(args, arg_name):
            overrides[field_name] = getattr(args, arg_name)
    return overrides or None


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    data_root = getattr(args, "data_root", Path("data"))

    path_overrides: dict | None = None
    if hasattr(args, "subject") or hasattr(args, "hemisphere"):
        subject = getattr(args, "subject", "subj01")
        hemisphere = getattr(args, "hemisphere", "lh")
        path_overrides = asdict(nsd_paths(data_root, subject=subject, hemisphere=hemisphere))
    else:
        path_overrides = _section_overrides(
            ExperimentPaths,
            args,
            {
                "fmri_path": "fmri_path",
                "checkpoint_path": "checkpoint",
                "analysis_cache_path": "analysis_cache",
                "rsa_heatmap_path": "heatmap",
            },
        )

    return experiment_config(
        data_root=data_root,
        paths=path_overrides,
        train=_section_overrides(
            TrainConfig,
            args,
            {
                "n_epochs": "epochs",
                "lr": "lr",
                "weight_decay": "weight_decay",
                "batch_size": "batch_size",
                "val_size": "val_size",
                "eval_every": "eval_every",
                "tau": "tau",
                "grad_clip": "grad_clip",
                "num_workers": "num_workers",
            },
        ),
        model=_section_overrides(
            ModelConfig,
            args,
            {"embed_dim": "embed_dim", "hidden_dim": "hidden_dim", "dropout": "dropout"},
        ),
        embeddings=_section_overrides(
            EmbeddingsConfig,
            args,
            {"clip_model_id": "clip_model"},
        ),
        analysis=_section_overrides(
            AnalysisConfig,
            args,
            {"rsa_bootstrap": "rsa_bootstrap", "pca_components": "pca_components"},
        ),
        run_ridge=False if getattr(args, "no_ridge", False) else None,
    )


def resolve_device(args: argparse.Namespace) -> torch.device:
    if hasattr(args, "device"):
        return torch.device(args.device)
    return default_device()


def stages_from_args(args: argparse.Namespace) -> PipelineStages:
    if args.all:
        return PipelineStages.all(plot=args.plot)
    if not (args.embeddings or args.train or args.analyze or args.plot):
        raise SystemExit(
            "Specify --all or at least one of --embeddings, --train, --analyze, --plot."
        )
    return PipelineStages(
        embeddings=args.embeddings,
        train=args.train,
        analyze=args.analyze,
        plot=args.plot,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    stages = stages_from_args(args)
    device = resolve_device(args)

    if stages.embeddings and not (stages.train or stages.analyze or stages.plot):
        compute_embeddings(config)
        return

    run_pipeline(config, stages, device=device)


if __name__ == "__main__":
    main()
