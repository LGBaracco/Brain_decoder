from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar

from brain_decoder.utils.device import default_device

__all__ = [
    "AnalysisConfig",
    "EmbeddingsConfig",
    "ExperimentConfig",
    "ExperimentPaths",
    "ModelConfig",
    "PipelineStages",
    "TrainConfig",
    "default_device",
    "experiment_config",
    "nsd_paths",
    "nsd_subj01_paths",
]


@dataclass(frozen=True)
class ExperimentPaths:
    data_root: Path = Path("data")
    fmri_path: Path = Path(
        "data/train_data/subj01/training_split/training_fmri/lh_training_fmri.npy"
    )
    train_embeddings_path: Path = Path("data/clip_embeddings/train_vitl14.npy")
    train_images_dir: Path = Path(
        "data/train_data/subj01/training_split/training_images"
    )
    test_images_dir: Path | None = Path("data/train_data/subj01/test_split/test_images")
    test_embeddings_path: Path | None = Path("data/clip_embeddings/test_vitl14.npy")
    checkpoint_path: Path = Path("data/encoders/best_encoder.pt")
    analysis_cache_path: Path = Path("data/analysis/val_analysis.npz")
    rsa_heatmap_path: Path = Path("data/figures/rsa_heatmap.png")


def nsd_paths(
    data_root: Path | str = "data",
    *,
    subject: str = "subj01",
    hemisphere: str = "lh",
) -> ExperimentPaths:
    root = Path(data_root)
    subj = root / "train_data" / subject
    fmri_name = f"{hemisphere}_training_fmri.npy"
    return ExperimentPaths(
        data_root=root,
        fmri_path=subj / "training_split/training_fmri" / fmri_name,
        train_embeddings_path=root / "clip_embeddings/train_vitl14.npy",
        train_images_dir=subj / "training_split/training_images",
        test_images_dir=subj / "test_split/test_images",
        test_embeddings_path=root / "clip_embeddings/test_vitl14.npy",
        checkpoint_path=root / "encoders/best_encoder.pt",
        analysis_cache_path=root / "analysis/val_analysis.npz",
        rsa_heatmap_path=root / "figures/rsa_heatmap.png",
    )


def nsd_subj01_paths(data_root: Path | str = "data") -> ExperimentPaths:
    return nsd_paths(data_root, subject="subj01", hemisphere="lh")


@dataclass(frozen=True)
class PipelineStages:
    embeddings: bool = False
    train: bool = False
    analyze: bool = False
    plot: bool = False

    @classmethod
    def all(cls, *, plot: bool = False) -> "PipelineStages":
        return cls(embeddings=True, train=True, analyze=True, plot=plot)

    def any_selected(self) -> bool:
        return self.embeddings or self.train or self.analyze or self.plot


@dataclass(frozen=True)
class TrainConfig:
    n_epochs: int = 50
    lr: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 512
    val_size: int = 500
    eval_every: int = 5
    grad_clip: float = 1.0
    tau: float = 0.07
    num_workers: int = 4


@dataclass(frozen=True)
class ModelConfig:
    embed_dim: int = 768
    hidden_dim: int = 512
    dropout: float = 0.3


@dataclass(frozen=True)
class EmbeddingsConfig:
    clip_model_id: str = "openai/clip-vit-large-patch14"


@dataclass(frozen=True)
class AnalysisConfig:
    rsa_bootstrap: int = 1000
    pca_components: int = 50


@dataclass(frozen=True)
class ExperimentConfig:
    paths: ExperimentPaths
    train: TrainConfig = TrainConfig()
    model: ModelConfig = ModelConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()
    analysis: AnalysisConfig = AnalysisConfig()
    run_ridge: bool = True

    @classmethod
    def default(cls, data_root: Path | str = "data") -> "ExperimentConfig":
        return cls(paths=nsd_subj01_paths(data_root))


_T = TypeVar("_T")


def _replace_section(current: _T, overrides: dict[str, Any]) -> _T:
    valid = {f.name for f in fields(current)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"Unknown fields {unknown} for {type(current).__name__}")
    return replace(current, **{k: overrides[k] for k in overrides if k in valid})


def experiment_config(
    *,
    paths: dict[str, Any] | None = None,
    train: dict[str, Any] | None = None,
    model: dict[str, Any] | None = None,
    embeddings: dict[str, Any] | None = None,
    analysis: dict[str, Any] | None = None,
    run_ridge: bool | None = None,
    data_root: Path | str = "data",
) -> ExperimentConfig:
    """Build config from defaults with per-section dict overrides."""
    cfg = ExperimentConfig.default(data_root)
    if paths is not None:
        cfg = replace(cfg, paths=_replace_section(cfg.paths, paths))
    if train is not None:
        cfg = replace(cfg, train=_replace_section(cfg.train, train))
    if model is not None:
        cfg = replace(cfg, model=_replace_section(cfg.model, model))
    if embeddings is not None:
        cfg = replace(cfg, embeddings=_replace_section(cfg.embeddings, embeddings))
    if analysis is not None:
        cfg = replace(cfg, analysis=_replace_section(cfg.analysis, analysis))
    if run_ridge is not None:
        cfg = replace(cfg, run_ridge=run_ridge)
    return cfg
