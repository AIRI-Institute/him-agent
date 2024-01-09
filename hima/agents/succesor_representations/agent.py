#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

from collections import deque
from enum import Enum, auto

import numpy as np

from hima.common.sdr import sparse_to_dense
from hima.common.utils import softmax, safe_divide
from hima.common.smooth_values import SSValue
from hima.modules.belief.cortial_column.cortical_column import CorticalColumn
from hima.modules.baselines.srtd import SRTD
from hima.modules.baselines.lstm import to_numpy, TLstmLayerHiddenState
from copy import copy
import torch
from hima.modules.belief.utils import EPS


class ExplorationPolicy(Enum):
    SOFTMAX = 1
    EPS_GREEDY = auto()


class SrEstimatePlanning(Enum):
    UNIFORM = 1
    ON_POLICY = auto()
    OFF_POLICY = auto()


class ActionValueEstimate(Enum):
    PREDICT = 1
    PLAN = auto()
    BALANCE = auto()


class BioHIMA:
    def __init__(
            self,
            cortical_column: CorticalColumn,
            *,
            gamma: float = 0.99,
            observation_reward_lr: float = 0.01,
            striatum_lr: float = 1.0,
            sr_steps: int = 5,
            approximate_tail: bool = True,
            inverse_temp: float = 1.0,
            exploration_eps: float = -1,
            action_value_estimate: str = 'plan',
            sr_estimate_planning: str = 'uniform',
            sr_early_stop: bool = True,
            lr_surprise=(0.2, 0.01),
            lr_td_error=(0.2, 0.01),
            adaptive_sr: bool = True,
            adaptive_lr: bool = True,
            srtd: SRTD | None,
            seed: int | None,
    ):
        self.observation_reward_lr = observation_reward_lr
        self.max_striatum_lr = striatum_lr
        self.cortical_column = cortical_column
        self.srtd = srtd
        self.gamma = gamma
        self.max_sr_steps = sr_steps
        self.sr_early_stop = sr_early_stop
        self.approximate_tail = approximate_tail
        self.inverse_temp = inverse_temp
        self.adaptive_sr = adaptive_sr
        self.adaptive_lr = adaptive_lr

        if exploration_eps < 0:
            self.exploration_policy = ExplorationPolicy.SOFTMAX
            self.exploration_eps = 0
        else:
            self.exploration_policy = ExplorationPolicy.EPS_GREEDY
            self.exploration_eps = exploration_eps

        self.action_value_estimate = ActionValueEstimate[action_value_estimate.upper()]
        self.sr_estimate_planning = SrEstimatePlanning[sr_estimate_planning.upper()]

        layer = self.cortical_column.layer
        layer_obs_size = layer.n_obs_states * layer.n_obs_vars
        layer_hidden_size = layer.n_hidden_states * layer.n_hidden_vars

        observation_rewards = np.zeros((layer.n_obs_vars, layer.n_obs_states))
        self.observation_rewards = observation_rewards.flatten()
        self.observation_messages = np.zeros_like(self.observation_rewards)

        # state backups for model-free TD
        self.previous_state = self.cortical_column.layer.internal_forward_messages.copy()
        self.previous_observation = np.zeros_like(self.observation_rewards)

        self.striatum_weights = np.zeros((layer_hidden_size, layer_obs_size))

        self.state_snapshot_stack = deque()

        self.predicted_sr = None
        self.generated_sr = None

        # metrics
        self.ss_td_error = SSValue(*lr_td_error)
        self.ss_surprise = SSValue(*lr_surprise)

        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self, initial_context_message, initial_external_message):
        assert len(self.state_snapshot_stack) == 0
        self.cortical_column.reset(initial_context_message, initial_external_message)

        self.previous_state = self.cortical_column.layer.internal_forward_messages.copy()
        self.previous_observation = np.zeros_like(self.observation_rewards)

    def sample_action(self):
        """Evaluate and sample actions."""
        action_values = self.evaluate_actions(with_planning=True)
        action_dist = self._get_action_selection_distribution(action_values, on_policy=True)
        action = self._rng.choice(self.n_actions, p=action_dist)
        return action

    def observe(self, observation, learn=True):
        """
        Main learning routine
            observation: tuple (image, action)
                events: sparse sdr
                action: sparse sdr
        """
        events, action = observation

        if events is not None:
            # predict current events using observed action
            self.cortical_column.observe(events, action, learn=learn)
            encoded_obs = self.cortical_column.output_sdr.sparse
            self.observation_messages = sparse_to_dense(encoded_obs, like=self.observation_messages)
        else:
            return

        if not learn:
            return

        # striatum TD learning
        self.predicted_sr, self.generated_sr, td_error = self.td_update_sr()

        self.ss_td_error.update(td_error)
        self.ss_surprise.update(self.cortical_column.surprise)

        return self.predicted_sr, self.generated_sr

    def reinforce(self, reward):
        """
        Adapt prior distribution of observations according to external reward.
            reward: float
        """
        # learn with mse loss
        lr, messages = self.observation_reward_lr, self.observation_messages

        self.observation_rewards += lr * messages * (reward - self.observation_rewards)

    def evaluate_actions(self, *, with_planning: bool = False):
        """Evaluate Q[s,a] for each action."""
        self._make_state_snapshot()

        n_actions = self.cortical_column.layer.external_input_size
        action_values = np.zeros(n_actions)
        dense_action = np.zeros_like(action_values)

        estimate_strategy = self._get_action_value_estimate_strategy(with_planning)

        for action in range(n_actions):
            # hacky way to clean previous one-hot, for 0-th does nothing
            dense_action[action - 1] = 0
            # set current one-hot
            dense_action[action] = 1

            self.cortical_column.predict(
                context_messages=self.cortical_column.layer.context_messages,
                external_messages=dense_action
            )

            if estimate_strategy == ActionValueEstimate.PLAN:
                sr = self.generate_sr(
                    self.sr_steps,
                    initial_messages=self.cortical_column.layer.prediction_cells,
                    initial_prediction=self.cortical_column.layer.prediction_columns,
                    save_state=False,
                    early_stop=self.sr_early_stop
                )
            else:
                sr = self.predict_sr(self.cortical_column.layer.prediction_cells)

            # average value predicted by all variables
            action_values[action] = np.sum(
                sr * self.observation_rewards
            ) / self.cortical_column.layer.n_obs_vars

            self._restore_last_snapshot(pop=False)

        self.state_snapshot_stack.pop()
        return action_values

    def generate_sr(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            save_state=True,
            return_predictions=False,
            early_stop=False
    ):
        """
            n_steps: number of prediction steps. If n_steps is 0 and approximate_tail is True,
            then this function is equivalent to predict_sr.
        """
        predictions = []

        if save_state:
            self._make_state_snapshot()

        sr = np.zeros_like(self.observation_messages)

        context_messages = initial_messages
        predicted_observation = initial_prediction

        discount = 1.0
        for t in range(n_steps):
            if early_stop:
                uni_dkl = (
                        np.log(self.cortical_column.layer.n_obs_states) +
                        np.sum(
                            predicted_observation * np.log(
                                np.clip(
                                    predicted_observation, EPS, None
                                )
                            )
                        )
                )

                if uni_dkl < EPS:
                    break

            sr += predicted_observation * discount

            if self.sr_estimate_planning == SrEstimatePlanning.UNIFORM:
                action_dist = None
            else:
                # on/off-policy
                # NB: evaluate actions directly with prediction, not with n-step planning!
                action_values = self.evaluate_actions(with_planning=False)
                action_dist = self._get_action_selection_distribution(
                    action_values,
                    on_policy=self.sr_estimate_planning == SrEstimatePlanning.ON_POLICY
                )

            self.cortical_column.predict(context_messages, external_messages=action_dist)

            context_messages = self.cortical_column.layer.internal_forward_messages.copy()
            predicted_observation = self.cortical_column.layer.prediction_columns

            discount *= self.gamma

            if return_predictions:
                predictions.append(copy(predicted_observation))

        if approximate_tail:
            sr += self.predict_sr(context_messages) * discount

        if save_state:
            self._restore_last_snapshot()

        if return_predictions:
            return sr, predictions
        else:
            return sr

    def predict_sr(self, hidden_vars_dist):
        if self.srtd is None:
            sr = np.dot(hidden_vars_dist, self.striatum_weights)
            sr /= self.cortical_column.layer.n_hidden_vars
        else:
            msg = torch.tensor(hidden_vars_dist).float().to(self.srtd.device)
            sr = to_numpy(self.srtd.predict_sr(msg, target=True))
        return sr

    def td_update_sr(self):
        target_sr = self.generate_sr(
            self.sr_steps,
            initial_messages=self.cortical_column.layer.internal_forward_messages,
            initial_prediction=self.observation_messages,
            approximate_tail=self.approximate_tail,
            early_stop=self.sr_early_stop
        )

        if self.srtd is None:
            predicted_sr = self.predict_sr(self.cortical_column.layer.internal_forward_messages)
            prediction_cells = self.current_state
            error_sr = target_sr - predicted_sr

            # dSR / dW for linear model
            delta_w = np.outer(prediction_cells, error_sr)

            self.striatum_weights += self.striatum_lr * delta_w
            self.striatum_weights = np.clip(self.striatum_weights, 0, None)

            td_error = np.mean(np.power(error_sr, 2))
        else:
            current_state = torch.tensor(self.current_state).float().to(self.srtd.device)
            predicted_sr = self.srtd.predict_sr(current_state, target=False)
            target_sr = torch.tensor(target_sr)
            target_sr = target_sr.float().to(self.srtd.device)

            td_error = self.srtd.compute_td_loss(
                target_sr,
                predicted_sr
            )
            predicted_sr = to_numpy(predicted_sr)
            target_sr = to_numpy(target_sr)

        return predicted_sr, target_sr, td_error

    @property
    def current_state(self):
        return self.cortical_column.layer.internal_forward_messages

    def _get_action_selection_distribution(
            self, action_values, on_policy: bool = True
    ) -> np.ndarray:
        # off policy means greedy, on policy — with current exploration strategy
        if on_policy and self.exploration_policy == ExplorationPolicy.SOFTMAX:
            # normalize values before applying softmax to make the choice
            # of the softmax temperature scale invariant
            action_values = safe_divide(action_values, np.abs(action_values.sum()))
            action_dist = softmax(action_values, beta=self.inverse_temp)
        else:
            # greedy off policy or eps-greedy
            best_action = np.argmax(action_values)
            # make greedy policy
            # noinspection PyTypeChecker
            action_dist = sparse_to_dense([best_action], like=action_values)

            if on_policy and self.exploration_policy == ExplorationPolicy.EPS_GREEDY:
                # add uniform exploration
                action_dist[best_action] = 1 - self.exploration_eps
                action_dist[:] += self.exploration_eps / self.n_actions

        return action_dist

    def _get_action_value_estimate_strategy(self, with_planning: bool) -> ActionValueEstimate:
        if not with_planning or self.action_value_estimate == ActionValueEstimate.PREDICT:
            return ActionValueEstimate.PREDICT

        # with_planning == True
        if self.action_value_estimate == ActionValueEstimate.PLAN or self._should_plan():
            return ActionValueEstimate.PLAN

        return ActionValueEstimate.PREDICT

    def _should_plan(self):
        p = np.clip(self.ss_td_error.norm_value, 0, 1)
        return self._rng.random() < p

    def _make_state_snapshot(self):
        self.state_snapshot_stack.append(
            self.cortical_column.make_state_snapshot()
        )

    def _restore_last_snapshot(self, pop: bool = True):
        snapshot = self.state_snapshot_stack.pop() if pop else self.state_snapshot_stack[-1]
        self.cortical_column.restore_last_snapshot(snapshot)

    @property
    def striatum_lr(self):
        if self.adaptive_lr:
            lr = self.max_striatum_lr * (1 - np.clip(self.ss_surprise.norm_value, 0, 1))
        else:
            lr = self.max_striatum_lr
        return lr

    @property
    def relative_log_surprise(self):
        s_base_log = np.log(np.log(self.cortical_column.layer.n_obs_states))
        s_current = np.clip(self.ss_surprise.current_value, 1e-7, None)
        s_min = np.clip(self.ss_surprise.mean - self.ss_surprise.std, 1e-7, None)
        return (1 - np.log(s_current) / s_base_log) / (1 - np.log(s_min) / s_base_log)

    @property
    def sr_steps(self):
        if self.adaptive_sr:
            sr_steps = int(
                max(
                    1,
                    np.round(
                        self.max_sr_steps *
                        np.clip(self.ss_td_error.norm_value, 0, 1) *
                        self.relative_log_surprise
                    )
                )
            )
        else:
            sr_steps = self.max_sr_steps
        return sr_steps

    @property
    def n_actions(self):
        return self.cortical_column.layer.external_input_size

    @property
    def surprise(self):
        return self.ss_surprise.current_value

    @property
    def td_error(self):
        return self.ss_td_error.current_value


class LstmBioHima(BioHIMA):
    """Patch-like adaptation of BioHIMA to work with LSTM layer."""

    def __init__(
            self, cortical_column: CorticalColumn,
            **kwargs
    ):
        super().__init__(cortical_column, **kwargs)

    # noinspection PyMethodOverriding
    def generate_sr(
            self,
            n_steps,
            initial_messages,
            initial_prediction,
            approximate_tail=True,
            save_state=True,
            return_predictions=False,
            early_stop=False
    ):
        """
            n_steps: number of prediction steps. If n_steps is 0 and approximate_tail is True,
            then this function is equivalent to predict_sr.
        """
        predictions = []

        if save_state:
            self._make_state_snapshot()

        sr = np.zeros_like(self.observation_messages)

        # represent state after getting observation
        context_messages = initial_messages
        # represent predicted observation
        predicted_observation = initial_prediction

        discount = 1.0
        for t in range(n_steps):
            if early_stop:
                uni_dkl = (
                        np.log(self.cortical_column.layer.n_obs_states) +
                        np.sum(
                            predicted_observation * np.log(
                                np.clip(
                                    predicted_observation, EPS, None
                                )
                            )
                        )
                )

                if uni_dkl < EPS:
                    break

            sr += predicted_observation * discount

            if self.sr_estimate_planning == SrEstimatePlanning.UNIFORM:
                action_dist = None
            else:
                # on/off-policy
                # NB: evaluate actions directly with prediction, not with n-step planning!
                action_values = self.evaluate_actions(with_planning=False)
                action_dist = self._get_action_selection_distribution(
                    action_values,
                    on_policy=self.sr_estimate_planning == SrEstimatePlanning.ON_POLICY
                )

            self.cortical_column.predict(context_messages, external_messages=action_dist)
            predicted_observation = self.cortical_column.layer.prediction_columns

            # THE ONLY ADDED CHANGE TO HIMA: explicitly observe predicted_observation
            self.cortical_column.layer.observe(predicted_observation, learn=False)
            # setting context is needed for action evaluation further on
            self.cortical_column.layer.set_context_messages(
                self.cortical_column.layer.internal_forward_messages
            )
            # ======

            context_messages = self.cortical_column.layer.internal_forward_messages.copy()
            discount *= self.gamma

            if return_predictions:
                predictions.append(copy(predicted_observation))

        if approximate_tail:
            sr += self.predict_sr(context_messages) * discount

        if save_state:
            self._restore_last_snapshot()

        # sr /= self.cortical_column.layer.n_hidden_vars

        if return_predictions:
            return sr, predictions
        else:
            return sr

    def _extract_state_from_context(self, context_messages: TLstmLayerHiddenState):
        # extract model state from layer state
        state_out, _ = context_messages[1]

        # convert model hidden state to probabilities
        state_probs_out = self.cortical_column.layer.model.to_probabilistic_out_state(state_out)
        return state_probs_out.detach()

    def predict_sr(self, context_messages: TLstmLayerHiddenState):
        msg = to_numpy(self._extract_state_from_context(context_messages))
        return super().predict_sr(msg)

    @property
    def current_state(self):
        return to_numpy(self._extract_state_from_context(
            self.cortical_column.layer.internal_forward_messages
        ))
