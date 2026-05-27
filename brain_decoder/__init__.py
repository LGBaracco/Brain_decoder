"""Brain decoding library."""

from brain_decoder.analysis import (
    extract_embeddings,
    load_analysis,
    run_rsa,
    save_analysis,
    save_rsa_heatmaps,
)
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
    nsd_subj01_paths,
)
from brain_decoder.data import NSDSubjectSource, PairedBrainDataset
from brain_decoder.embeddings import CLIPImageExtractor, save_image_embeddings
from brain_decoder.losses import infonce_loss
from brain_decoder.models import BrainEncoder, ridge_baseline
from brain_decoder.run import (
    compute_embeddings,
    load_encoder,
    plot_analysis,
    run_analysis,
    run_pipeline,
    train_encoder,
)
from brain_decoder.training import ContrastiveTrainer, evaluate_encoder
from brain_decoder.utils import cosine_rdm, default_device, retrieval_accuracy, upper_tri

__all__ = [
    "AnalysisConfig",
    "BrainEncoder",
    "CLIPImageExtractor",
    "ContrastiveTrainer",
    "EmbeddingsConfig",
    "ExperimentConfig",
    "ExperimentPaths",
    "ModelConfig",
    "NSDSubjectSource",
    "PairedBrainDataset",
    "PipelineStages",
    "TrainConfig",
    "compute_embeddings",
    "cosine_rdm",
    "default_device",
    "evaluate_encoder",
    "experiment_config",
    "extract_embeddings",
    "infonce_loss",
    "load_analysis",
    "load_encoder",
    "nsd_paths",
    "nsd_subj01_paths",
    "plot_analysis",
    "retrieval_accuracy",
    "ridge_baseline",
    "run_analysis",
    "run_pipeline",
    "run_rsa",
    "save_analysis",
    "save_image_embeddings",
    "save_rsa_heatmaps",
    "train_encoder",
    "upper_tri",
]
