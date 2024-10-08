#  Copyright (c) 2024 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np
import socket
import json
import atexit
import pygraphviz as pgv
import colormap
from hima.modules.belief.utils import get_data, send_string, NumpyEncoder, normalize

HOST = "127.0.0.1"
PORT = 5555
EPS = 1e-24


def sparse_to_dense(sparse, size):
    dense = np.zeros(size)
    dense[sparse] = 1
    return dense


class ToyDHTM:
    """
        Simplified, fully deterministic DHTM
        for one hidden variable with visualizations.
        Stores transition matrix explicitly.
    """
    vis_server: socket.socket = None

    def __init__(
            self,
            n_obs_states,
            n_actions,
            n_clones,
            gamma: float = 0.99,
            visualize: bool = False,
            visualization_server=(HOST, PORT)
    ):
        self.n_clones = n_clones
        self.n_obs_states = n_obs_states
        self.n_actions = n_actions
        self.n_hidden_states = self.n_clones * self.n_obs_states
        self.gamma = gamma
        self.visualize = visualize
        self.vis_server_address = visualization_server

        self.transition_counts = np.zeros(
            (self.n_actions, self.n_hidden_states, self.n_hidden_states),
            dtype=np.int32
        )
        self.prior = np.zeros(self.n_hidden_states, dtype=np.float64)
        self.activation_counts = np.zeros(self.n_hidden_states, dtype=np.float64)

        self.observation_buffer = list()
        self.action_buffer = list()
        self.state_buffer = list()
        self.belief_state = None

        self.vis_server = None
        if self.visualize:
            self.connect_to_vis_server()
            if self.vis_server is not None:
                atexit.register(self.close)

    def reset(self, gridworld_map):
        if len(self.state_buffer) > 0:
            self.prior[self.state_buffer[0]] += 1

        self.belief_state = None
        self.clear_buffers()
        if self.vis_server is not None:
            self._send_events([('reset', {'gridworld_map': gridworld_map})])

    def replay(self):
        if len(self.observation_buffer) == 0:
            return 0.0

        total_surprise = 0
        self.belief_state = None
        p_action = None
        for obs_state, action in zip(self.observation_buffer, self.action_buffer):
            if self.belief_state is None:
                prediction = sparse_to_dense(self.state_buffer[0], self.n_hidden_states)
                self.belief_state = prediction
            else:
                prediction = self.belief_state @ self.transition_counts[p_action]
                self.belief_propagation(obs_state, p_action)

            prediction = prediction.reshape(-1, self.n_clones).sum(axis=-1)
            prediction = normalize(prediction).flatten()

            surprise = - np.log(
                np.clip(prediction[obs_state], EPS, 1.0)
            )
            total_surprise += surprise
            p_action = action

        return total_surprise / len(self.observation_buffer)

    def clear_buffers(self):
        self.observation_buffer.clear()
        self.action_buffer.clear()
        self.state_buffer.clear()

    def predict(self):
        if self.belief_state is None:
            prediction = self.prior
        else:
            action = self.action_buffer[-1]
            prediction = self.belief_state @ self.transition_counts[action]

        prediction = prediction.reshape(-1, self.n_clones).sum(axis=-1)
        prediction = normalize(prediction).flatten()
        return prediction

    def belief_propagation(self, obs_state, action):
        # belief propagation
        column_states = self._get_column_states(obs_state)
        if self.belief_state is None:
            belief_state = np.zeros(self.n_hidden_states)
            belief_state[column_states] = self.prior[column_states]
            self.belief_state = normalize(belief_state).flatten()
        else:
            prediction = self.belief_state @ self.transition_counts[action]
            belief_state = np.zeros(self.n_hidden_states)
            belief_state[column_states] = prediction[column_states]
            self.belief_state = normalize(belief_state).flatten()

    def observe(self, obs_state, action, true_pos=None):
        # for debugging
        # event type: (name: str, data: tuple)
        events = list()

        if len(self.action_buffer) > 0:
            p_action = self.action_buffer[-1]
        else:
            p_action = None

        self.belief_propagation(obs_state, p_action)

        self.observation_buffer.append(obs_state)
        self.action_buffer.append(action)
        # state to be defined
        self.state_buffer.append(None)

        step = len(self.observation_buffer) - 1
        pos = step
        resolved = False

        events.append(('new_true_pos', true_pos))
        events.append(('new_obs', pos, obs_state, action))

        while not resolved:
            if step == 0:
                # initial step
                column_states = self._get_column_states(obs_state)
                state = self._get_maximum_prior_state(column_states, pos)
                self.state_buffer[pos] = state
                resolved = True

                events.append(('set_state', self._state_to_clone(state)))
            else:
                # input variables
                obs_state = self.observation_buffer[pos]
                column_states = self._get_column_states(obs_state)
                state = self.state_buffer[pos]

                prev_state = self.state_buffer[pos - 1]
                prev_action = self.action_buffer[pos - 1]

                prediction = self.transition_counts[prev_action, prev_state].flatten()
                sparse_prediction = np.flatnonzero(prediction)

                if state is None:
                    coincide = np.isin(sparse_prediction, column_states)
                else:
                    coincide = np.isin(sparse_prediction, state)

                correct_prediction = sparse_prediction[coincide]
                wrong_prediction = sparse_prediction[~coincide]

                events.append(
                    (
                        'predict_forward',
                        [
                            self._state_to_clone(x, return_obs_state=True) + (w,)
                            for x, w in
                            zip(correct_prediction, prediction[correct_prediction])
                        ],
                        [
                            self._state_to_clone(x, return_obs_state=True) + (w,)
                            for x, w in
                            zip(wrong_prediction, prediction[wrong_prediction])
                        ]
                    )
                )
                # cases:
                # 1. correct set is not empty
                if len(correct_prediction) > 0:
                    state = self._get_best_prediction(prediction, correct_prediction)
                    self.state_buffer[pos] = state
                    resolved = True

                    events.append(('set_state', self._state_to_clone(state)))
                # 2. correct set is empty
                else:
                    if len(wrong_prediction) == 0:
                        if state is None:
                            state = self._get_maximum_prior_state(column_states, pos)
                            self.state_buffer[pos] = state

                            events.append(('set_state', self._state_to_clone(state)))

                        resolved = True
                    else:
                        # resampling previous clone
                        # try to use backward connections first
                        if state is None:
                            prediction = self.transition_counts[
                                prev_action, :, column_states
                            ].sum(axis=0).flatten()
                        else:
                            prediction = self.transition_counts[
                                prev_action, :, state
                            ].flatten()

                            # punish connection
                            self.activation_counts[prev_state] -= 1
                            if (pos - 2) >= 0:
                                pp_action = self.action_buffer[pos - 2]
                                pp_state = self.state_buffer[pos - 2]
                                assert self.transition_counts[pp_action, pp_state, prev_state] > 0
                                self.transition_counts[pp_action, pp_state, prev_state] -= 1

                                events.append(
                                    (
                                        'punish_con',
                                        pp_action,
                                        self._state_to_clone(pp_state, return_obs_state=True),
                                        self._state_to_clone(prev_state, return_obs_state=True)
                                    )
                                )

                                if self.transition_counts[pp_action, pp_state, prev_state] == 0:
                                    events.append(
                                        (
                                            'remove_con',
                                            pp_action,
                                            self._state_to_clone(pp_state, return_obs_state=True),
                                            self._state_to_clone(prev_state, return_obs_state=True)
                                        )
                                    )

                        sparse_prediction = np.flatnonzero(prediction)
                        prev_obs_state = self.observation_buffer[pos - 1]

                        prev_column_states = self._get_column_states(prev_obs_state)
                        coincide = np.isin(sparse_prediction, prev_column_states)
                        correct_prediction = sparse_prediction[coincide]
                        wrong_prediction = sparse_prediction[~coincide]

                        events.append(
                            (
                                'predict_backward',
                                [
                                    self._state_to_clone(x, return_obs_state=True) + (w,)
                                    for x, w in
                                    zip(correct_prediction, prediction[correct_prediction])
                                ],
                                [
                                    self._state_to_clone(x, return_obs_state=True) + (w,)
                                    for x, w in
                                    zip(wrong_prediction, prediction[wrong_prediction])
                                ]
                            )
                        )

                        if len(correct_prediction) > 0:
                            prev_state = self._get_best_prediction(prediction, correct_prediction)
                            if state is None:
                                prediction = self.transition_counts[prev_action, prev_state].flatten()
                                sparse_prediction = np.flatnonzero(prediction)
                                coincide = np.isin(sparse_prediction, column_states)
                                correct_prediction = sparse_prediction[coincide]
                                state = self._get_best_prediction(prediction, correct_prediction)
                                self.state_buffer[pos] = state

                                events.append(('set_state', self._state_to_clone(state)))
                        else:
                            # choose the least used clone
                            prev_column_states = prev_column_states[
                                np.argsort(
                                    self.activation_counts[prev_column_states]
                                )
                            ][::-1]
                            scores = np.zeros(len(prev_column_states))
                            for i, ps in enumerate(prev_column_states):
                                prediction = self.transition_counts[
                                    prev_action, ps].flatten()

                                scores[i] = prediction.sum()
                                if len(sparse_prediction) == 0:
                                    prev_state = ps
                                    break
                            else:
                                prev_state = prev_column_states[
                                    np.argmin(scores)
                                ]

                            if state is None:
                                state = self._get_maximum_prior_state(column_states, pos)
                                self.state_buffer[pos] = state

                                events.append(('set_state', self._state_to_clone(state)))

                        self.state_buffer[pos - 1] = prev_state

                        events.append(
                            ('set_prev_state', self._state_to_clone(prev_state))
                        )

                # in any case
                self.transition_counts[prev_action, prev_state, state] += 1
                self.activation_counts[column_states] *= self.gamma
                self.activation_counts[state] += 1

                events.append(
                    (
                        'reinforce_con',
                        prev_action,
                        self._state_to_clone(prev_state, return_obs_state=True),
                        self._state_to_clone(state, return_obs_state=True)
                    )
                )
                # move to previous position
                if not resolved:
                    pos -= 1

                    events.append(('move', pos))

                    if pos == 0:
                        resolved = True

            if self.vis_server is not None:
                self._send_events(events)

            events.clear()

    def _get_maximum_prior_state(self, candidates, pos):
        state = None
        if pos > 0:
            if np.isin(self.state_buffer[pos-1], candidates):
                state = self.state_buffer[pos-1]
        if state is None:
            state = candidates[
                np.argmax(self.activation_counts[candidates])
            ]
        return state

    def _get_best_prediction(self, prediction, candidates):
        return candidates[
            np.argmax(
                prediction[candidates]
            )
        ]

    def _state_to_clone(self, state, return_obs_state=False):
        obs_state = state // self.n_clones
        clone = state - self.n_clones * obs_state
        if return_obs_state:
            return clone, obs_state
        else:
            return clone

    def _get_column_states(self, obs_state):
        return np.arange(self.n_clones) + obs_state * self.n_clones

    def connect_to_vis_server(self):
        self.vis_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        try:
            self.vis_server.connect(self.vis_server_address)
            # handshake
            self._send_json_dict({'type': 'hello'})
            data = get_data(self.vis_server)
            print(data)

            if data != 'toy_dhtm':
                raise socket.error(
                    f'Handshake failed {self.vis_server_address}: It is not ToyDHTM vis server!'
                )
            print(f'Connected to visualization server {self.vis_server_address}!')
        except socket.error as msg:
            self.vis_server.close()
            self.vis_server = None
            print(f'Failed to connect to the visualization server: {msg}. Proceed.')

    def close(self):
        if self.vis_server is not None:
            self._send_json_dict({'type': 'close'})
            self.vis_server.close()
            print('Connection closed.')
        try:
            atexit.unregister(self.close)
        except Exception as e:
            print("exception unregistering close method", e)

    def _send_events(self, events):
        data = get_data(self.vis_server)
        if data == 'skip':
            self._send_json_dict({'type': 'skip'})
        elif data == 'close':
            self.vis_server.close()
            self.vis_server = None
            print('Server shutdown. Proceed.')
        elif data == 'step':
            data_dict = {
                'type': 'events',
                'events': events
            }
            self._send_json_dict(data_dict)

    def _send_json_dict(self, data_dict):
        send_string(json.dumps(data_dict, cls=NumpyEncoder), self.vis_server)

    def draw_graph(self, path=None, connection_threshold=0, activation_threshold=0, labels=None):
        g = pgv.AGraph(strict=False, directed=True)
        outline_color = '#3655b3'
        nonzero_states = np.flatnonzero(self.activation_counts > activation_threshold)
        node_cmap = colormap.cmap_builder('Pastel1')
        edge_cmap = colormap.Colormap().cmap_bicolor('white', 'blue')

        for state in nonzero_states:
            if labels is not None:
                label = str(labels[state]) + '_'
            else:
                label = ''
            clone, obs_state = self._state_to_clone(state, return_obs_state=True)
            g.add_node(
                f'{label}{obs_state}({clone})',
                style='filled',
                fillcolor=colormap.rgb2hex(
                    *(node_cmap(obs_state / self.n_obs_states)[:-1]),
                    normalised=True
                ),
                color=outline_color
            )

        for u in nonzero_states:
            if labels is not None:
                u_label = str(labels[u]) + '_'
            else:
                u_label = ''

            u_clone, u_obs_state = self._state_to_clone(u, return_obs_state=True)
            for action in range(self.n_actions):
                transitions = self.transition_counts[action, u]
                weights = transitions / (transitions.sum() + EPS)
                nonzero_transitions = np.flatnonzero(weights > connection_threshold)

                for v, weight in zip(nonzero_transitions, weights[nonzero_transitions]):
                    if self.activation_counts[v] > activation_threshold:
                        if labels is not None:
                            v_label = str(labels[v]) + '_'
                        else:
                            v_label = ''
                        v_clone, v_obs_state = self._state_to_clone(v, return_obs_state=True)
                        line_color = colormap.rgb2hex(
                            *(edge_cmap(int(255 * weight))[:-1]),
                            normalised=True
                        )
                        g.add_edge(
                            f'{u_label}{u_obs_state}({u_clone})', f'{v_label}{v_obs_state}({v_clone})',
                            color=line_color,
                            label=str(action)
                        )

        g.layout(prog='dot')
        return g.draw(path, format='png')
