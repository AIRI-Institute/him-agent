#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Optional, Any

import numpy as np
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from wandb.sdk.wandb_run import Run

from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.metrics import mean_absolute_error, kl_div
from hima.experiments.temporal_pooling.utils import rename_dict_keys


class ExperimentStats:
    blocks: dict[str, Any]
    sequence_ids_order: list[int]
    sequences_block_stats: dict[int, dict[str, Any]]

    def __init__(self, blocks):
        self.blocks = blocks
        self.sequences_block_stats = {}
        self.sequence_ids_order = []

    def on_new_sequence(self, sequence_id: int):
        if sequence_id == self.current_sequence_id:
            return
        self.sequence_ids_order.append(sequence_id)
        self.sequences_block_stats[sequence_id] = {}

        for block_name in self.blocks:
            block = self.blocks[block_name]
            block.reset_stats()
            self.current_block_stats[block.name] = block.stats

    @property
    def current_sequence_id(self):
        return self.sequence_ids_order[-1] if self.sequence_ids_order else None

    @property
    def previous_sequence_id(self):
        return self.sequence_ids_order[-2] if len(self.sequence_ids_order) >= 2 else None

    @property
    def current_block_stats(self):
        return self.sequences_block_stats[self.current_sequence_id]

    def on_step(self, logger):
        if logger is None:
            return

        metrics = {}
        for block_name in self.current_block_stats:
            block = self.blocks[block_name]
            block_stats = self.current_block_stats[block_name]
            block_metrics = block_stats.get_metrics()
            block_metrics = rename_dict_keys(block_metrics, add_prefix=f'{block.tag}/')
            metrics |= block_metrics

        logger.log(metrics)

    def on_finish(self, logger: Run):
        if not logger:
            return

        to_log, to_sum = {}, {}
        logger.log(to_log)
        for key, val in to_sum.items():
            logger.summary[key] = val

    # noinspection PyProtectedMember
    def _get_tp_metrics(self, temporal_pooler) -> dict:
        prev_repr = self.tp_current_representation
        curr_repr_lst = temporal_pooler.getUnionSDR().sparse
        curr_repr = set(curr_repr_lst)
        self.tp_current_representation = curr_repr
        # noinspection PyTypeChecker
        self.last_representations[self.current_sequence_id] = curr_repr

        self.tp_sequence_total_trials[self.current_sequence_id] += 1
        cluster_trials = self.tp_sequence_total_trials[self.current_sequence_id]

        output_distribution_counts = self.tp_output_distribution_counts[self.current_sequence_id]
        output_distribution_counts[curr_repr_lst] += 1
        cluster_size = np.count_nonzero(output_distribution_counts)
        cluster_distribution = output_distribution_counts / cluster_trials

        step_sparsity = safe_divide(
            len(curr_repr), self.tp_output_sdr_size
        )
        step_relative_sparsity = safe_divide(
            len(curr_repr), self.tp_expected_active_size
        )
        new_cells_relative_ratio = safe_divide(
            len(curr_repr - prev_repr), self.tp_expected_active_size
        )
        sym_diff_cells_ratio = safe_divide(
            len(curr_repr ^ prev_repr),
            len(curr_repr | prev_repr)
        )
        step_metrics = {
            'tp/step/sparsity': step_sparsity,
            'tp/step/relative_sparsity': step_relative_sparsity,
            'tp/step/new_cells_relative_ratio': new_cells_relative_ratio,
            'tp/step/sym_diff_cells_ratio': sym_diff_cells_ratio,
        }

        cluster_sparsity = safe_divide(
            cluster_size, self.tp_output_sdr_size
        )
        cluster_relative_sparsity = safe_divide(
            cluster_size, self.tp_expected_active_size
        )
        cluster_binary_active_coverage = safe_divide(
            len(curr_repr), cluster_size
        )
        cluster_distribution_active_coverage = cluster_distribution[curr_repr_lst].sum()
        cluster_distribution_active_coverage /= self.tp_expected_active_size

        cluster_entropy = self._cluster_entropy(cluster_distribution)
        cluster_entropy_active_coverage = safe_divide(
            self._cluster_entropy(cluster_distribution[curr_repr_lst]),
            cluster_entropy
        )
        sequence_metrics = {
            'tp/sequence/sparsity': cluster_sparsity,
            'tp/sequence/relative_sparsity': cluster_relative_sparsity,
            'tp/sequence/cluster_binary_coverage': cluster_binary_active_coverage,
            'tp/sequence/cluster_distribution_coverage': cluster_distribution_active_coverage,
            'tp/sequence/entropy': cluster_entropy,
            'tp/sequence/entropy_coverage': cluster_entropy_active_coverage,
        }
        return step_metrics | sequence_metrics

    def _cluster_kl_div(self, x: np.ndarray, y: np.ndarray) -> float:
        h = kl_div(x, y)
        h /= self.tp_expected_active_size
        h /= np.log(self.tp_output_sdr_size)
        return h

    def _cluster_entropy(self, x: np.ndarray) -> float:
        return self._cluster_kl_div(x, x)

    def _get_tm_metrics(self, temporal_memory) -> dict:
        active_cells: np.ndarray = temporal_memory.get_active_cells()
        predicted_cells: np.ndarray = temporal_memory.get_correctly_predicted_cells()

        recall = safe_divide(predicted_cells.size, active_cells.size)

        return {
            'tm/recall': recall
        }

    def _get_summary_new_sdrs(self, policies):
        n_policies = len(policies)
        diag_mask = np.identity(n_policies, dtype=bool)

        input_similarity_matrix = self._get_input_similarity(policies)
        input_similarity_matrix = np.ma.array(input_similarity_matrix, mask=diag_mask)

        output_similarity_matrix = self._get_output_similarity(self.last_representations)
        output_similarity_matrix = np.ma.array(output_similarity_matrix, mask=diag_mask)

        input_similarity_matrix = standardize_distr(input_similarity_matrix)
        output_similarity_matrix = standardize_distr(output_similarity_matrix)

        smae = mean_absolute_error(input_similarity_matrix, output_similarity_matrix)

        representation_similarity_plot = self._plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.ma.abs(output_similarity_matrix - input_similarity_matrix)
        )
        to_log = {
            'representations_similarity_sdr': representation_similarity_plot,
        }
        to_sum = {
            'standardized_mae_sdr': smae,
        }
        return to_log, to_sum

    def _get_summary_old_actions(self, policies):
        n_policies = len(policies)
        diag_mask = np.identity(n_policies, dtype=bool)

        input_similarity_matrix = self._get_policy_action_similarity(policies)
        input_similarity_matrix = np.ma.array(input_similarity_matrix, mask=diag_mask)

        output_similarity_matrix = self._get_output_similarity_union(
            self.last_representations
        )
        output_similarity_matrix = np.ma.array(output_similarity_matrix, mask=diag_mask)

        unnorm_representation_similarity_plot = self._plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix
        )

        input_similarity_matrix = standardize_distr(input_similarity_matrix)
        output_similarity_matrix = standardize_distr(output_similarity_matrix)

        smae = mean_absolute_error(input_similarity_matrix, output_similarity_matrix)

        representation_similarity_plot = self._plot_similarity_matrices(
            input=input_similarity_matrix,
            output=output_similarity_matrix,
            diff=np.ma.abs(output_similarity_matrix - input_similarity_matrix)
        )
        to_log = {
            'raw_representations_similarity': unnorm_representation_similarity_plot,
            'representations_similarity': representation_similarity_plot,
        }
        to_sum = {
            'standardized_mae': smae,
        }
        return to_log, to_sum

    def _get_summary_old_actions_distr(self, policies):
        n_policies = len(policies)
        diag_mask = np.identity(n_policies, dtype=bool)

        input_similarity_matrix = self._get_policy_action_similarity(policies)
        input_similarity_matrix = np.ma.array(input_similarity_matrix, mask=diag_mask)

        output_similarity_matrix = self._get_output_similarity_distr()
        output_similarity_matrix = np.ma.array(output_similarity_matrix, mask=diag_mask)

        unnorm_input_similarity_matrix = input_similarity_matrix
        input_similarity_matrix = standardize_distr(input_similarity_matrix)

        unnorm_output_similarity_matrix = output_similarity_matrix
        output_similarity_matrix = standardize_distr(output_similarity_matrix)

        representation_similarity_plot = self._plot_similarity_matrices(
            raw_input_sim=unnorm_input_similarity_matrix,
            raw_output_kl_div=unnorm_output_similarity_matrix,
            input_sim=input_similarity_matrix,
            output_kl_div=output_similarity_matrix
        )
        to_log = {
            'representations_kl_div': representation_similarity_plot,
        }
        return to_log

    def _get_policy_action_similarity(self, policies):
        n_policies = len(policies)
        similarity_matrix = np.zeros((n_policies, n_policies))

        for i in range(n_policies):
            for j in range(n_policies):

                counter = 0
                size = 0
                for p1, p2 in zip(policies[i], policies[j]):
                    _, a1 = p1
                    _, a2 = p2

                    size += 1
                    # such comparison works only for bucket encoding
                    if a1[0] == a2[0]:
                        counter += 1

                similarity_matrix[i, j] = safe_divide(counter, size)
        return similarity_matrix

    def _get_input_similarity(self, policies):
        def elem_sim(x1, x2):
            overlap = np.intersect1d(x1, x2, assume_unique=True).size
            return safe_divide(overlap, x2.size)

        def reduce_elementwise_similarity(similarities):
            return np.mean(similarities)

        n_policies = len(policies)
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                similarities = []
                for p1, p2 in zip(policies[i], policies[j]):
                    p1_sim = [elem_sim(p1[k], p2[k]) for k in range(len(p1))]
                    sim = reduce_elementwise_similarity(p1_sim)
                    similarities.append(sim)

                similarity_matrix[i, j] = reduce_elementwise_similarity(similarities)
        return similarity_matrix

    def _get_output_similarity_union(self, representations):
        n_policies = len(representations.keys())
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                repr1: set = representations[i]
                repr2: set = representations[j]
                similarity_matrix[i, j] = safe_divide(
                    len(repr1 & repr2),
                    len(repr2 | repr2)
                )
        return similarity_matrix

    def _get_output_similarity(self, representations):
        n_policies = len(representations.keys())
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                repr1: set = representations[i]
                repr2: set = representations[j]
                similarity_matrix[i, j] = safe_divide(
                    len(repr1 & repr2),
                    len(repr2)
                )
        return similarity_matrix

    def _get_output_similarity_distr(self):
        n_policies = len(self.tp_output_distribution_counts.keys())
        similarity_matrix = np.zeros((n_policies, n_policies))
        for i in range(n_policies):
            for j in range(n_policies):
                if i == j:
                    continue

                distr1 = self.tp_output_distribution_counts[i] / self.tp_sequence_total_trials[i]
                distr2 = self.tp_output_distribution_counts[j] / self.tp_sequence_total_trials[j]

                similarity_matrix[i, j] = self._cluster_kl_div(distr1, distr2)
        return similarity_matrix

    def _plot_similarity_matrices(self, **sim_matrices):
        n = len(sim_matrices)
        heatmap_size = 6
        fig, axes = plt.subplots(
            nrows=1, ncols=n, sharey='all',
            figsize=(heatmap_size * n, heatmap_size)
        )

        for ax, (name, sim_matrix) in zip(axes, sim_matrices.items()):
            vmin = 0 if np.min(sim_matrix) >= 0 else -1
            if isinstance(sim_matrix, np.ma.MaskedArray):
                sns.heatmap(
                    sim_matrix, mask=sim_matrix.mask,
                    vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True
                )
            else:
                sns.heatmap(sim_matrix, vmin=vmin, vmax=1, cmap='plasma', ax=ax, annot=True)
            ax.set_title(name, size=10)

        return wandb.Image(axes[0])

    def _get_final_representations(self):
        n_clusters = len(self.last_representations)
        representations = np.zeros((n_clusters, self.tp_output_sdr_size), dtype=float)
        distributions = np.zeros_like(representations)

        for i, policy_id in enumerate(self.last_representations.keys()):
            repr = self.last_representations[policy_id]
            distr = self.tp_output_distribution_counts[policy_id]
            trials = self.tp_sequence_total_trials[policy_id]

            representations[i, list(repr)] = 1.
            distributions[i] = distr / trials

        fig, ax1 = plt.subplots(1, 1, figsize=(16, 8))
        sns.heatmap(representations, vmin=0, vmax=1, cmap='plasma')

        fig, ax2 = plt.subplots(1, 1, figsize=(16, 8))
        sns.heatmap(distributions, vmin=0, vmax=1, cmap='plasma', ax=ax2)

        return {
            'representations': wandb.Image(ax1),
            'distributions': wandb.Image(ax2)
        }


def standardize_distr(x: np.ndarray) -> np.ndarray:
    return (x - np.mean(x)) / (np.max(x) - np.min(x))
