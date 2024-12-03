#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import jit
from numpy.random import Generator

from hima.common.scheduler import Scheduler
from hima.experiments.temporal_pooling.stp.sp import SpNewbornPruningMode
from hima.experiments.temporal_pooling.stp.se_utils import nb_choice_k


class PruningController:
    owner: Any

    mode: SpNewbornPruningMode
    n_stages: int
    stage: int
    scheduler: Scheduler

    initial_rf_sparsity: float
    target_rf_to_input_ratio: float
    target_rf_sparsity: float

    def __init__(
            self, owner,
            mode: str, cycle: float, n_stages: int,
            target_rf_sparsity: float = None, target_rf_to_input_ratio: float = None
    ):
        self.owner = owner

        # noinspection PyTypeChecker
        self.mode = SpNewbornPruningMode[mode.upper()]
        self.n_stages = n_stages
        self.stage = 0

        schedule = int(cycle / owner.output_sds.sparsity)
        self.scheduler = Scheduler(schedule)

        self.initial_rf_sparsity = 1.0
        self.target_rf_to_input_ratio = target_rf_to_input_ratio
        self.target_rf_sparsity = target_rf_sparsity

    @property
    def is_newborn_phase(self):
        return self.stage < self.n_stages

    def shrink_receptive_field(self, pruned_mask):
        self.stage += 1

        if self.mode == SpNewbornPruningMode.LINEAR:
            new_sparsity = self.newborn_linear_progress(
                initial=self.initial_rf_sparsity, target=self.get_target_rf_sparsity()
            )
        elif self.mode == SpNewbornPruningMode.POWERLAW:
            new_sparsity = self.newborn_powerlaw_progress(
                initial=self.owner.rf_sparsity, target=self.get_target_rf_sparsity()
            )
        else:
            raise ValueError(f'Pruning mode {self.mode} is not supported')

        if new_sparsity > self.owner.rf_sparsity:
            # if feedforward sparsity is tracked, then it may change and lead to RF increase
            # ==> leave RF as is
            return

        # sample what connections to keep for each neuron independently
        new_rf_size = round(new_sparsity * self.owner.ff_size)

        prune(
            self.owner.rng, self.owner.weights, self.owner.weights_pow_p,
            new_rf_size, pruned_mask
        )
        return new_sparsity, new_rf_size

    def get_target_rf_sparsity(self):
        if self.target_rf_sparsity is not None:
            return self.target_rf_sparsity

        if self.owner.adapt_to_ff_sparsity:
            ff_sparsity = self.owner.ff_avg_sparsity
        else:
            ff_sparsity = self.owner.feedforward_sds.sparsity

        return self.target_rf_to_input_ratio * ff_sparsity

    def newborn_linear_progress(self, initial, target):
        newborn_phase_progress = self.stage / self.n_stages
        # linear decay rule
        return initial + newborn_phase_progress * (target - initial)

    # noinspection PyUnusedLocal
    def newborn_powerlaw_progress(self, initial, target):
        steps_left = self.n_stages - self.stage + 1
        current = self.owner.rf_sparsity
        # what decay is needed to reach the target in the remaining steps
        # NB: recalculate each step to exclude rounding errors
        decay = np.power(target / current, 1 / steps_left)
        # exponential decay rule
        return current * decay


@jit()
def prune(
        rng: Generator, weights: npt.NDArray[float], pow_weights: npt.NDArray[float],
        k: int, pruned_mask
):
    # WARNING: works only with non-negative weights!
    n_neurons, n_synapses = weights.shape
    w_priority = weights if pow_weights is None else pow_weights

    for row in range(n_neurons):
        pm_row = pruned_mask[row]
        w_row = weights[row]
        w_priority_row = w_priority[row]

        active_mask = ~pm_row
        prune_probs = pruning_probs_from_synaptic_weights(w_priority_row[active_mask])

        # pruned connections are marked as already selected for "select K from N" operation
        n_active = len(prune_probs)
        not_k = n_active - k
        ixs = nb_choice_k(rng, not_k, prune_probs, n_active, False)
        new_pruned_ixs = np.flatnonzero(active_mask)[ixs]
        w_row[new_pruned_ixs] = 0.0
        pm_row[new_pruned_ixs] = True
        if pow_weights is not None:
            pow_weights[row][new_pruned_ixs] = 0.0


@jit()
def pruning_probs_from_synaptic_weights(weights):
    priority = weights.copy()
    # normalize relative to the mean: < 1 are weak, > 1 are strong
    priority /= priority.mean()
    # clip to avoid numerical issues: low values threshold is safe to keep enough information
    #   i.e. we keep info until the synapse is 1mln times weaker than the average
    np.clip(priority, 1e-6, 1e+6, priority)
    # linearize the scales -> [-X, +Y], where X,Y are low < 100
    priority = np.log(priority)
    # -> shift to negative [-(X+Y), 0] -> flip to positive [0, X+Y] -> add baseline probability
    #   the weakest synapses are now have the highest probability
    prune_probs = -(priority - priority.max()) + 0.1
    return prune_probs
