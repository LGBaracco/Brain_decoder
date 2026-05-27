# Brain decoder

Contrastive brain→CLIP decoding on NSD fMRI ([dataset](https://algonautsproject.com/2023/braindata.html), [CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)).

## Setup

```bash
uv sync
```

## Quick start (CLI)

```bash
uv run brain-decoder --all --plot
uv run brain-decoder --embeddings
uv run brain-decoder --analyze --plot
uv run brain-decoder --train --lr 1e-3 --epochs 20
```

Run `uv run brain-decoder --help` for all flags (grouped: data, train, model, embeddings, analysis).

## Customize in Python

Import what you need and change only the section you care about.

```python
from brain_decoder import (
    PipelineStages,
    experiment_config,
    run_pipeline,
)

config = experiment_config(train={"lr": 1e-3, "batch_size": 256})
run_pipeline(config, PipelineStages(train=True, analyze=True))
```

Start from defaults and use `dataclasses.replace` for full control:

```python
from dataclasses import replace
from brain_decoder import ExperimentConfig, PipelineStages, run_pipeline

config = ExperimentConfig.default()
config = replace(config, train=replace(config.train, lr=1e-4, n_epochs=30))
run_pipeline(config, PipelineStages(plot=True))
```

### Run stages separately

```python
from brain_decoder import (
    ExperimentConfig,
    NSDSubjectSource,
    compute_embeddings,
    train_encoder,
    run_analysis,
    plot_analysis,
)

config = ExperimentConfig.default()
compute_embeddings(config)

source = NSDSubjectSource(config.paths.fmri_path, config.paths.train_embeddings_path)
encoder = train_encoder(source, config)
run_analysis(encoder, source, config)
plot_analysis(config)  # reads cache only
```

### Building blocks

```python
from brain_decoder import (
    BrainEncoder,
    CLIPImageExtractor,
    ContrastiveTrainer,
    infonce_loss,
    run_rsa,
    retrieval_accuracy,
)
```

See [`examples/custom_run.py`](examples/custom_run.py).

## Parameter reference

| Field | Default | Stage |
|-------|---------|--------|
| `paths.data_root` | `data` | all |
| `paths.fmri_path` | subj01 lh training fMRI | train, analyze |
| `paths.train_embeddings_path` | `clip_embeddings/train_vitl14.npy` | train, analyze |
| `paths.checkpoint_path` | `encoders/best_encoder.pt` | train, analyze |
| `paths.analysis_cache_path` | `analysis/val_analysis.npz` | analyze, plot |
| `paths.rsa_heatmap_path` | `figures/rsa_heatmap.png` | plot |
| `train.lr` | `3e-4` | train |
| `train.n_epochs` | `50` | train |
| `train.batch_size` | `512` | train, analyze |
| `train.val_size` | `500` | train, analyze |
| `train.tau` | `0.07` | train |
| `model.hidden_dim` | `512` | train, analyze |
| `model.embed_dim` | `768` | train, analyze |
| `model.dropout` | `0.3` | train |
| `embeddings.clip_model_id` | `openai/clip-vit-large-patch14` | embeddings |
| `analysis.rsa_bootstrap` | `1000` | analyze |
| `analysis.pca_components` | `50` | plot (embedding dims per stimulus; heatmap axes are N validation stimuli, default 500) |
| `run_ridge` | `true` | train |

Subject layout: `nsd_paths("data", subject="subj02", hemisphere="rh")` or CLI `--subject subj02 --hemisphere rh`.

## Recipes

**Different subject**

```python
from brain_decoder import experiment_config, nsd_paths, PipelineStages, run_pipeline

paths = nsd_paths("data", subject="subj02")
from dataclasses import asdict
config = experiment_config(paths=asdict(paths))
run_pipeline(config, PipelineStages(train=True))
```

**Replot without re-running analyze**

```bash
uv run brain-decoder --plot --pca-components 30
```

**Skip ridge baseline**

```python
experiment_config(run_ridge=False)
# or: uv run brain-decoder --train --no-ridge
```
