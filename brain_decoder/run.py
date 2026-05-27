from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from brain_decoder.analysis.extract import extract_embeddings
from brain_decoder.analysis.rsa import run_rsa
from brain_decoder.analysis.visualize import save_analysis, save_rsa_heatmaps
from brain_decoder.config import ExperimentConfig, PipelineStages
from brain_decoder.data.dataset import PairedBrainDataset
from brain_decoder.data.nsd import NSDSubjectSource
from brain_decoder.embeddings.clip_hf import CLIPImageExtractor
from brain_decoder.embeddings.pipeline import save_image_embeddings
from brain_decoder.models.baselines import ridge_baseline
from brain_decoder.models.encoders import BrainEncoder
from brain_decoder.training.trainer import ContrastiveTrainer
from brain_decoder.utils.device import default_device


def data_source_from_config(config: ExperimentConfig) -> NSDSubjectSource:
    paths = config.paths
    if not paths.train_embeddings_path.is_file():
        raise FileNotFoundError(
            f"Training embeddings not found: {paths.train_embeddings_path}\n"
            "Run the embeddings stage first (`--embeddings` or `compute_embeddings`)."
        )
    if not paths.fmri_path.is_file():
        raise FileNotFoundError(f"fMRI data not found: {paths.fmri_path}")
    return NSDSubjectSource(paths.fmri_path, paths.train_embeddings_path)


def build_loaders(
    split,
    *,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader]:
    train_ds = PairedBrainDataset(split.fmri_train, split.target_train)
    val_ds = PairedBrainDataset(split.fmri_val, split.target_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def compute_embeddings(config: ExperimentConfig) -> None:
    paths = config.paths
    extractor = CLIPImageExtractor(
        model_id=config.embeddings.clip_model_id,
    )
    save_image_embeddings(paths.train_images_dir, paths.train_embeddings_path, extractor)
    if paths.test_images_dir is not None and paths.test_embeddings_path is not None:
        save_image_embeddings(
            paths.test_images_dir,
            paths.test_embeddings_path,
            extractor,
        )


def load_encoder(
    config: ExperimentConfig,
    data_source: NSDSubjectSource,
    device: torch.device,
) -> BrainEncoder:
    path = config.paths.checkpoint_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Run the train stage first (`--train` or `train_encoder`)."
        )
    model = config.model
    encoder = BrainEncoder(
        n_voxels=data_source.n_voxels,
        embed_dim=model.embed_dim,
        hidden_dim=model.hidden_dim,
        dropout=model.dropout,
    ).to(device)
    encoder.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return encoder


def train_encoder(
    data_source: NSDSubjectSource,
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
) -> BrainEncoder:
    device = device or default_device()
    train = config.train
    model = config.model
    ckpt = config.paths.checkpoint_path

    split = data_source.train_val_split(train.val_size)
    if config.run_ridge:
        ridge_baseline(split.fmri_train, split.target_train, split.fmri_val, split.target_val)

    train_loader, val_loader = build_loaders(
        split,
        batch_size=train.batch_size,
        num_workers=train.num_workers,
    )

    encoder = BrainEncoder(
        n_voxels=data_source.n_voxels,
        embed_dim=model.embed_dim,
        hidden_dim=model.hidden_dim,
        dropout=model.dropout,
    ).to(device)

    print(f"\nEncoder: {sum(p.numel() for p in encoder.parameters()):,} parameters")
    print(f"Voxels: {data_source.n_voxels} | Device: {device}")
    print(f"Expected random-chance loss: {np.log(train.batch_size):.3f}\n")

    trainer_config = replace(train, checkpoint_path=ckpt)
    ContrastiveTrainer(encoder, train_loader, val_loader, device, trainer_config).fit()
    return encoder


def run_analysis(
    encoder: BrainEncoder,
    data_source: NSDSubjectSource,
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
) -> tuple[float, tuple[float, float]]:
    device = device or default_device()
    train = config.train
    analysis = config.analysis

    split = data_source.train_val_split(train.val_size)
    _, val_loader = build_loaders(
        split,
        batch_size=train.batch_size,
        num_workers=train.num_workers,
    )

    z_brain, z_target = extract_embeddings(encoder, val_loader, device)
    rsa_r, (ci_low, ci_high) = run_rsa(
        z_brain,
        z_target,
        n_bootstrap=analysis.rsa_bootstrap,
    )
    save_analysis(
        config.paths.analysis_cache_path,
        z_brain,
        z_target,
        rsa_r,
        ci_low,
        ci_high,
    )
    return rsa_r, (ci_low, ci_high)


def plot_analysis(config: ExperimentConfig) -> Path:
    paths = config.paths
    return save_rsa_heatmaps(
        paths.analysis_cache_path,
        paths.rsa_heatmap_path,
        n_components=config.analysis.pca_components,
    )


def run_pipeline(
    config: ExperimentConfig,
    stages: PipelineStages,
    *,
    device: torch.device | None = None,
) -> BrainEncoder | None:
    if not stages.any_selected():
        raise ValueError("No pipeline stages selected.")

    device = device or default_device()

    if stages.embeddings:
        compute_embeddings(config)

    encoder: BrainEncoder | None = None
    needs_data = stages.train or stages.analyze

    if needs_data:
        data_source = data_source_from_config(config)
        if stages.train:
            encoder = train_encoder(data_source, config, device=device)
        if stages.analyze:
            if encoder is None:
                encoder = load_encoder(config, data_source, device)
            run_analysis(encoder, data_source, config, device=device)

    if stages.plot:
        plot_analysis(config)

    return encoder
