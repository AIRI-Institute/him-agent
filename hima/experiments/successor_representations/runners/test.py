#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import pickle
import sys
import os
from typing import Union, Any

import numpy as np
from hima.common.config.base import read_config, override_config
from hima.common.run.argparse import parse_arg_list
from hima.common.sdr import sparse_to_dense
from hima.experiments.successor_representations.runners.base import BaseRunner
from hima.modules.belief.utils import normalize


class ICMLRunner(BaseRunner):
    @staticmethod
    def make_agent(agent_type, conf):
        if agent_type == 'bio':
            from hima.experiments.successor_representations.runners.agents\
                import BioAgentWrapper
            agent = BioAgentWrapper(conf)
        elif agent_type == 'q':
            from hima.experiments.successor_representations.runners.agents\
                import QTableAgentWrapper
            agent = QTableAgentWrapper(conf)
        elif agent_type == 'sr':
            from hima.experiments.successor_representations.runners.agents\
                import SRTableAgentWrapper
            agent = SRTableAgentWrapper(conf)
        elif agent_type == 'data':
            from hima.experiments.successor_representations.runners.agents\
                import DatasetCreatorAgent
            agent = DatasetCreatorAgent(**conf)
        else:
            raise NotImplementedError

        return agent

    @staticmethod
    def make_environment(env_type, conf, setup):
        if env_type == 'pinball':
            from hima.experiments.successor_representations.runners.envs import PinballWrapper
            env = PinballWrapper(conf, setup)
        elif env_type == 'animalai':
            from hima.experiments.successor_representations.runners.envs import AnimalAIWrapper
            env = AnimalAIWrapper(conf, setup)
        elif env_type == 'gridworld':
            from hima.experiments.successor_representations.runners.envs import GridWorldWrapper
            env = GridWorldWrapper(conf, setup)
        else:
            raise NotImplementedError
        return env

    def switch_strategy(self, strategy):
        if strategy == 'random':
            self.reward_free = True
        elif strategy == 'non-random':
            self.reward_free = False

    def set_learning(self, memory, striatum):
        agent = self.agent.agent

        if striatum:
            if agent.max_striatum_lr == 0:
                agent.max_striatum_lr = self.max_striatum_learning_rate
        else:
            self.max_striatum_learning_rate = agent.max_striatum_lr
            agent.max_striatum_lr = 0

        if memory:
            if agent.cortical_column.layer.lr == 0:
                agent.cortical_column.layer.lr = self.lr
        else:
            self.lr = agent.cortical_column.layer.lr
            agent.cortical_column.layer.lr = 0

    @property
    def obs_reward(self):
        agent = self.agent.agent
        decoder = agent.cortical_column.decoder
        if decoder is not None:
            obs_rewards = decoder.decode(
                normalize(
                    agent.rewards.reshape(
                        agent.cortical_column.layer.n_obs_vars, -1
                    )
                ).flatten()
            ).reshape(self.environment.raw_obs_shape)
        else:
            obs_rewards = agent.rewards.reshape(
                agent.cortical_column.layer.n_obs_vars, -1
            )
        return obs_rewards

    @property
    def clean_surprise(self):
        layer = self.agent.agent.cortical_column.layer
        return self.agent.agent.surprise * layer.is_any_segment_active

    @property
    def real_reward(self):
        im = sparse_to_dense(self.agent.events, shape=self.environment.raw_obs_shape)
        real_reward = im * self.reward
        return real_reward

    @property
    def encoded_reward(self):
        im = self.encoder_output
        return im * self.reward

    @property
    def state_visited(self):
        env = self.environment.environment
        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        values[r, c] = 1

        return values, 1

    @property
    def state_value(self):
        env = self.environment.environment

        r, c = env.r, env.c
        values = np.zeros((env.h, env.w))
        state_value = self.agent.state_value
        values[r, c] = state_value

        counts = np.zeros_like(values)
        counts[r, c] = 1
        return values, counts

    @property
    def state_representation(self):
        internal_messages = self.agent.agent.cortical_column.layer.internal_messages
        cells_per_column = self.agent.agent.cortical_column.layer.cells_per_column
        return internal_messages.reshape(-1, cells_per_column).T

    @property
    def q_value(self):
        env = self.environment.environment
        # left, right, up, down
        actions = self.environment.actions
        shifts = np.array([[0, 0], [0, env.w], [env.h, 0], [env.h, env.w]])

        r, c = env.r, env.c
        values = np.zeros((env.h * 2, env.w * 2))
        action_values = self.agent.action_values
        counts = np.zeros_like(values)

        for value, shift in zip(action_values, shifts):
            x, y = r + shift[0], c + shift[1]
            values[x, y] = value
            counts[x, y] = 1

        return values, counts

    def get_true_sr(self, path=None, observation_radius=-1):
        """
            Save SR/SF formed by table-sr agent

            observation_radius: if greater than -1, converts SR to SF
            for corresponding observation window

            Feature indexing for observation radius=1:

            0 1 2
            3 4 5 - center
            6 7 8
        """
        agent = self.agent
        env = self.environment.environment

        t = agent.sr
        t = np.mean(t, axis=0)

        if observation_radius >= 0:
            # convert sr to sf
            n_features = (2*observation_radius + 1)**2
            colors = env.unique_colors
            colors_map = env.colors

            if n_features == 1:
                colors_map = colors_map[1:-1, 1:-1]

            sfs = list()

            for feature in range(n_features):
                # form transformation matrix
                row_shift = feature // (2*observation_radius + 1)
                col_shift = feature % (2*observation_radius + 1)
                state_to_color = colors_map[
                    row_shift:row_shift+env.h,
                    col_shift:col_shift+env.w
                ].flatten()

                masks = list()
                for color in colors:
                    masks.append(state_to_color == color)

                masks = np.vstack(masks).T
                # (n_states, colors)
                sfs.append(np.dot(t, masks))

            t = np.hstack(sfs)

        if path is not None:
            np.save(path, t)

        return t

    @property
    def sr(self):
        agent = self.agent
        env = self.environment.environment

        t = agent.sr
        if len(t.shape) > 2:
            t = np.mean(t, axis=0)

        all_srs = list()
        for r in range(env.h):
            srs = list()
            for c in range(env.w):
                sr = t[r*env.w + c].reshape(env.h, env.w)
                srs.append(sr)
            all_srs.append(srs)

        return np.block(all_srs)

    @property
    def rewards(self):
        agent = self.agent.agent
        return agent.rewards.reshape(1, -1)

    @property
    def sf_diff(self):
        return np.mean(self.agent.predicted_sf - self.agent.planned_sf)

    @property
    def raw_observation(self):
        return np.dot(self.obs[:, :, :3], [299 / 1000, 587 / 1000, 114 / 1000])

    @property
    def camera_output(self):
        return sparse_to_dense(self.agent.events, shape=self.environment.raw_obs_shape)

    @property
    def encoder_output(self):
        return self.agent.agent.observation_messages.reshape(
            self.agent.agent.cortical_column.layer.n_obs_vars,
            -1
        )

    def save_encoder(self, path):
        with open(
            os.path.join(path,
             f'{self.logger.name}_{self.episodes}episodes_sp.pkl'),
            'wb'
        ) as file:
            pickle.dump(
            {
                    'encoder': self.agent.agent.cortical_column.encoder,
                    'camera': self.agent.camera,
                    'decoder': self.agent.agent.cortical_column.decoder
                },
                file=file
            )

    def load_encoder_state(self, file_path):
        with open(file_path, 'rb') as file:
            state_dict = pickle.load(file)

        self.agent.agent.cortical_column.encoder = state_dict['encoder']
        self.agent.agent.cortical_column.decoder = state_dict['decoder']
        self.agent.camera = state_dict['camera']

    def save_buffer(self, path):
        layer = self.agent.agent.cortical_column.layer
        cells_per_column = layer.cells_per_column

        if len(layer.observation_messages_buffer) == 0:
            return

        obs_buffer = np.array(layer.observation_messages_buffer)
        ext_buffer = np.array(layer.external_messages_buffer)
        fwd_buffer = np.array(layer.forward_messages_buffer)
        bwd_buffer = np.array(layer.backward_messages_buffer)
        prior_buffer = np.array([layer.forward_messages_buffer[0]] + layer.prediction_buffer)
        trajectory = np.array(
            [np.zeros_like(self.environment.trajectory[0])] + self.environment.trajectory
        )

        fwd_buffer = np.transpose(
            fwd_buffer.reshape(fwd_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        bwd_buffer = np.transpose(
            bwd_buffer.reshape(bwd_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        prior_buffer = np.transpose(
            prior_buffer.reshape(prior_buffer.shape[0], -1, cells_per_column),
            (0, 2, 1)
        )
        ext_buffer = ext_buffer.reshape(ext_buffer.shape[0], 1, -1)
        obs_buffer = obs_buffer.reshape(obs_buffer.shape[0], 1, -1)

        if self.logger is not None:
            run_name = self.logger.name
        else:
            run_name = str(self.seed)

        for name, array in zip(
                ['ext', 'obs', 'fwd', 'bwd', 'prior', 'traj'],
                [ext_buffer, obs_buffer, fwd_buffer, bwd_buffer, prior_buffer, trajectory]
        ):
            path_name = os.path.join(
                path,
                f'{run_name}_{self.episodes}_{name}.npy'
            )
            np.save(path_name, array)


def main(config_path):
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    config: dict[str, Union[Union[dict[str, Any], list[Any]], Any]] = dict()

    # main part
    config['run'] = read_config(config_path)

    env_conf_path = config['run'].pop('env_conf')
    config['env_type'] = env_conf_path.split('/')[-2]
    config['env'] = read_config(env_conf_path)

    agent_conf_path = config['run'].pop('agent_conf')
    config['agent_type'] = agent_conf_path.split('/')[-2]
    config['agent'] = read_config(agent_conf_path)

    metrics_conf = config['run'].pop('metrics_conf')
    if metrics_conf is not None:
        config['metrics'] = read_config(metrics_conf)

    if config['run']['seed'] is None:
        config['run']['seed'] = np.random.randint(0, np.iinfo(np.int32).max)

    # unfolding subconfigs
    if config['agent_type'] == 'bio':
        def load_subconfig(entity, conf):
            conf_path = config['agent'].pop(f'{entity}_conf')
            conf['agent'][f'{entity}_type'] = conf_path.split('/')[-2]
            conf['agent'][entity] = read_config(conf_path)

        load_subconfig('layer', config)

        if 'encoder_conf' in config['agent']:
            load_subconfig('encoder', config)
        else:
            config['agent']['encoder_type'] = None

        if 'decoder_conf' in config['agent']:
            load_subconfig('decoder', config)
        else:
            config['agent']['decoder_type'] = None

        if 'srtd_conf' in config['agent']:
            load_subconfig('srtd', config)
        else:
            config['agent']['srtd_type'] = None

        if 'striatum_conf' in config['agent']:
            load_subconfig('striatum', config)
        else:
            config['agent']['striatum_type'] = None

    elif config['agent_type'] == 'q':
        config['agent']['qvn'] = read_config(config['agent'].pop('qvn_conf'))

        if 'ucb_estimate_conf' in config['agent']:
            config['agent']['ucb_estimate'] = read_config(config['agent'].pop('ucb_estimate_conf'))
        else:
            config['agent']['ucb_estimate'] = None

        if 'eligibility_traces' in config['agent']:
            config['agent']['eligibility_traces'] = read_config(
                config['agent'].pop('eligibility_traces_conf'))
        else:
            config['agent']['eligibility_traces'] = None

    # override some values
    overrides = parse_arg_list(sys.argv[2:])
    override_config(config, overrides)

    if config['run'].pop('log'):
        import wandb
        logger = wandb.init(
            project=config['run'].pop('project_name'), entity=os.environ.get('WANDB_ENTITY', None),
            config=config
        )
    else:
        logger = None

    runner = ICMLRunner(logger, config)
    runner.run()


if __name__ == '__main__':
    default_config = 'configs/runner/pinball.yaml'
    main(os.environ.get('RUN_CONF', default_config))
