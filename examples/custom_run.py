"""Example: customize one section, run selected stages."""

from brain_decoder import PipelineStages, experiment_config, run_pipeline

# Change only training; everything else stays at defaults.
config = experiment_config(
    train={"lr": 1e-3, "n_epochs": 10},
    model={"hidden_dim": 256},
)

run_pipeline(config, PipelineStages(train=True, analyze=True, plot=True))
