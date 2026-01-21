"""
experiment.py
Experiment tracking and environment setup with proper config handling.
"""

import os
import time
import logging
from pathlib import Path
import random
from dataclasses import asdict
import torch
from typing import Optional
import wandb
import numpy as np
from .hypers import ExperimentHypers, Hypers    
from manabot.infra.profiler import Profiler
import manabot.infra.log

CODE_CONTEXT_ROOT = Path(os.getenv("CODE_CONTEXT_ROOT", str(Path.home() / "src")))

def flatten_config(cfg: dict, parent_key: str = '', sep: str = '/') -> dict:
    """Flatten nested dictionary with path-like keys."""
    items = []
    for k, v in cfg.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

class Experiment:
    """Experiment tracking and environment setup."""
    def __init__(self, experiment_hypers: ExperimentHypers = ExperimentHypers(), full_hypers: Hypers = Hypers()):
        self.experiment_hypers = experiment_hypers
        self.full_hypers = full_hypers
        self.exp_name = experiment_hypers.exp_name
        self.seed = experiment_hypers.seed
        self.torch_deterministic = experiment_hypers.torch_deterministic
        self.device = experiment_hypers.device
        self.wandb_on = experiment_hypers.wandb
        self.wandb_project_name = experiment_hypers.wandb_project_name
        self.runs_dir = self.experiment_hypers.runs_dir / self.exp_name
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.wandb_run = None
        self.profiler = Profiler(enabled=experiment_hypers.profiler_enabled)
        level = getattr(logging, self.experiment_hypers.log_level.upper(), logging.INFO)
        manabot.infra.log.setGlobalLogLevel(level)
        self.logger = manabot.infra.log.getLogger(__name__)

        self._setup_random()
        self._setup_tracking()

    def _get_flattened_config(self) -> dict:
        """Convert nested hypers to flat dict for wandb."""
        config_dict = {
            "observation": asdict(self.full_hypers.observation),
            "match": asdict(self.full_hypers.match),
            "train": asdict(self.full_hypers.train),
            "reward": asdict(self.full_hypers.reward),
            "agent": asdict(self.full_hypers.agent),
            "experiment": asdict(self.full_hypers.experiment),
        }
        return flatten_config(config_dict)

    def _setup_tracking(self):
        """Setup experiment tracking with wandb."""
        run_name = f"{self.exp_name}__{self.seed}__{int(time.time())}"
        run_dir = self.runs_dir / run_name
        
        if self.wandb_on:
            try:
                config = self._get_flattened_config()
                config.update({
                    "seed": self.seed,
                    "device": self.device,
                    "runs_dir": str(self.runs_dir),
                    "code_context": str(CODE_CONTEXT_ROOT),
                })
                self.wandb_run = wandb.init(
                    project=self.wandb_project_name,
                    entity=None,
                    name=run_name,
                    dir=str(self.runs_dir),
                    config=config,
                    monitor_gym=True,
                    save_code=True,
                )
                self.logger.info(f"Initialized wandb run: {run_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize wandb: {e}")
                self.wandb_on = False

    def log(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to wandb."""
        if self.wandb_on and self.wandb_run:
            self.wandb_run.log(metrics, step=step)
    
    def add_scalar(self, tag: str, value, step: int):
        """Compatibility method for tensorboard-style logging, now uses wandb."""
        if self.wandb_on and self.wandb_run:
            self.wandb_run.log({tag: value}, step=step)

    def _setup_random(self):
        """Setup random number generators."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = self.torch_deterministic

    def close(self):
        """Cleanup wandb resources."""
        if self.wandb_on and self.wandb_run:
            try:
                wandb.finish()
                self.wandb_run = None
            except Exception as e:
                self.logger.warning(f"Error finishing wandb run: {e}")

    # -------------------------------------------------------------------------
    # Performance Logging
    # -------------------------------------------------------------------------
    def log_performance(self, step: Optional[int] = None) -> None:
        """
        Logs hierarchical performance metrics. For each node in the hierarchy,
        logs the total time spent, its percentage of the parent's time, and its
        percentage of the overall (root) time. The keys are prefixed with
        'hierarchical/'.
        """
        if not self.wandb_on or not self.wandb_run:
            self.logger.warning("Wandb not initialized, skipping performance logging")
            return

        perf_stats = self.profiler.get_stats()

        log_dict = {}
        for node_path, data in perf_stats.items():
            log_dict[f"performance/{node_path}/pct_of_total"] = data["pct_of_total"]

        self.wandb_run.log(log_dict, step=step)
        self.logger.info(f"Performance stats: {log_dict}")
