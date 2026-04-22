# Copyright 2026 Thousand Brains Project
#
# Use of this source code is governed by the MIT license.
"""Pretraining experiment variants compatible with MuJoCo agent partials.

The base ``MontySupervisedObjectPretrainingExperiment.setup_experiment`` reads
``config["environment"]["env_init_args"]["agents"]["agent_args"]["positions"]``,
which assumes a Habitat-style ``agents`` dict. With MuJoCo configs ``agents``
is a ``functools.partial`` (Hydra ``_target_`` + ``_partial_: true``) and is
not subscriptable. This subclass skips that Habitat-specific lookup.
"""

from __future__ import annotations

import numpy as np

from tbp.monty.frameworks.experiments.monty_experiment import MontyExperiment
from tbp.monty.frameworks.experiments.pretraining_experiments import (
    MontySupervisedObjectPretrainingExperiment,
)


class MuJoCoMontySupervisedObjectPretrainingExperiment(
    MontySupervisedObjectPretrainingExperiment
):
    def setup_experiment(self, config):
        # Bypass MontySupervisedObjectPretrainingExperiment.setup_experiment, which
        # assumes a Habitat-style agents dict. Call its grandparent directly.
        MontyExperiment.setup_experiment(self, config)
        # sensor_pos is only meaningful for multi-LM Habitat layouts; with the
        # MuJoCo robot agent we have a single sensor patch, so a zero offset is
        # fine (matches the parent's else-branch fallback).
        self.sensor_pos = np.array([0, 0, 0])
