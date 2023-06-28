#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from matplotlib import pyplot as plt
from minisom import MiniSom

from hima.common.sds import Sds
from hima.envs.mnist import MNISTEnv
from hima.experiments.temporal_pooling.stp.sp import SpatialPooler


class SpAttractorMnistExperiment:
    def __init__(self, logger, conf):
        self.seed = conf['run']['seed']
        self.logger = logger

        self.env = MNISTEnv(seed=self.seed)

        input_sds = Sds(shape=self.env.obs_shape, sparsity=1.0)

        if conf.get('encoder') is not None:
            conf['encoder']['seed'] = self.seed
            conf['encoder']['feedforward_sds'] = input_sds
            conf['encoder']['output_sds'] = Sds(conf['encoder']['output_sds'])
            self.encoder = SpatialPooler(
                **conf['encoder']
            )

            conf['attractor']['adapt_to_ff_sparsity'] = False
            attractor_sds = self.encoder.output_sds
        else:
            conf['attractor']['adapt_to_ff_sparsity'] = True
            attractor_sds = input_sds
            self.encoder = None

        conf['attractor']['feedforward_sds'] = attractor_sds
        conf['attractor']['output_sds'] = Sds(conf['attractor']['output_sds'])
        conf['attractor']['seed'] = self.seed
        self.attractor = SpatialPooler(
            **conf['attractor']
        )

        if conf['run'].get('max_steps') is not None:
            self.max_steps = conf['run']['max_steps']
        else:
            self.max_steps = self.env.size

        self.n_episodes = conf['run']['n_episodes']
        self.log_update_rate = conf['run'].get('update_rate')
        self.n_trajectories = conf['run'].get('n_trajectories', 0)
        self.attractor_steps = conf['run'].get('attractor_steps', 0)
        self.learn_attractor = conf['run'].get('learn_attractor_in_loop', False)
        self.pairs_per_trajectory = conf['run'].get('pairs_per_trajectory', 1)

        self.som_iterations = conf['run'].get('som_iterations', 100)
        self.som_learning_rate = conf['run'].get('som_learning_rate', 0.5)
        self.som_sigma = conf['run'].get('som_sigma', 1.0)
        self.som_size = conf['run'].get('som_size', 100)

        if self.logger is not None:
            self.logger.define_metric(
                name='main_metrics/relative_similarity',
                step_metric='iteration'
            )
            self.logger.define_metric(
                name='convergence/io_hist',
                step_metric='iteration'
            )
            self.logger.define_metric(
                name='som/clusters',
                step_metric='iteration'
            )

            for cls in range(10):
                self.logger.define_metric(
                    name=f'relative_similarity/class {cls}',
                    step_metric='iteration'
                )

    def run(self):
        for i in range(self.n_episodes):
            steps = 0
            att_entropy = []
            enc_entropy = []

            self.env.reset()

            while True:
                obs = self.preprocess(self.env.obs())
                self.env.step()

                if self.encoder is not None:
                    obs = self.encoder.compute(obs, learn=True)
                    enc_entropy.append(self.encoder.output_entropy())

                self.attractor.compute(obs, learn=True)
                att_entropy.append(self.attractor.output_entropy())

                steps += 1
                if steps >= self.max_steps:
                    break

            if self.logger is not None:
                self.logger.log(
                    {'main_metrics/attractor_entropy': np.array(att_entropy).mean()},
                    step=i
                )
                if self.encoder is not None:
                    self.logger.log(
                        {'main_metrics/encoder_entropy': np.array(enc_entropy).mean()},
                        step=i
                    )

                if (self.log_update_rate is not None) and (i % self.log_update_rate == 0):
                    start_classes = list()
                    trajectories = list()

                    for _ in range(self.n_trajectories):
                        trajectory = list()

                        image, cls = self.env.obs(return_class=True)
                        self.env.step()

                        pattern = self.preprocess(image)
                        trajectory.append(pattern)

                        for _ in range(self.attractor_steps):
                            if (self.encoder is not None) and (_ == 0):
                                pattern = self.encoder.compute(pattern, learn=False)
                            else:
                                pattern = self.attractor.compute(pattern, self.learn_attractor)

                            trajectory.append(pattern)

                        trajectories.append(trajectory)
                        start_classes.append(cls)

                    similarities = list()
                    sim_matrices = np.zeros((self.attractor_steps+1, 10, 10))
                    class_counts = np.zeros((10, 10))

                    # generate non-repetitive trajectory pairs
                    pair1 = np.repeat(
                        np.arange(len(trajectories) - self.pairs_per_trajectory),
                        self.pairs_per_trajectory
                    )
                    pair2 = (
                        np.tile(
                            np.arange(self.pairs_per_trajectory) + 1,
                            len(trajectories) - self.pairs_per_trajectory
                        ) + pair1
                    )
                    for p1, p2 in zip(pair1, pair2):
                        similarity = list()
                        cls1 = start_classes[p1]
                        cls2 = start_classes[p2]
                        class_counts[cls1, cls2] += 1
                        class_counts[cls2, cls1] += 1

                        for att_step, x in enumerate(zip(trajectories[p1], trajectories[p2])):
                            sim = self.similarity(x[0], x[1])

                            sim_matrices[att_step, cls1, cls2] += sim
                            sim_matrices[att_step, cls2, cls1] += sim

                            similarity.append(sim)

                        similarities.append(similarity)

                    sim_matrices /= class_counts
                    # divide each row in each matrix by its diagonal element
                    rel_sim = pd.DataFrame(
                        (
                            sim_matrices / np.diagonal(sim_matrices, axis1=1, axis2=2)[:, :, None]
                        ).mean(axis=-1)
                    )

                    self.logger.log(
                        {
                            'convergence/relative_similarity': wandb.Image(
                                sns.lineplot(rel_sim)
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                    self.logger.log(
                        {
                            'convergence/class_pair_counts': wandb.Image(
                                sns.heatmap(class_counts)
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                    fig, axs = plt.subplots(ncols=4, sharey='row', figsize=(16, 4))
                    axs[0].set_title('raw_sim')
                    axs[1].set_title('1-step')
                    axs[2].set_title(f'{self.attractor_steps//2}-step')
                    axs[3].set_title(f'{self.attractor_steps}-step')
                    sns.heatmap(sim_matrices[0], ax=axs[0], cmap='viridis')
                    sns.heatmap(sim_matrices[1], ax=axs[1], cmap='viridis')
                    sns.heatmap(sim_matrices[self.attractor_steps//2], ax=axs[2], cmap='viridis')
                    sns.heatmap(sim_matrices[-1], ax=axs[3], cmap='viridis')

                    self.logger.log(
                        {
                            'convergence/similarity_per_class': wandb.Image(
                                fig
                            )
                        },
                        step=i
                    )
                    plt.close('all')

        if self.logger is not None:
            similarities = np.array(similarities)
            in_sim = similarities[:, 0]

            start_images = [x[0] for x in trajectories]
            dense_start_images = np.zeros(
                (len(start_images), self.encoder.feedforward_sds.size),
                dtype='float32'
            )
            for im_id, x in enumerate(start_images):
                dense_start_images[im_id, x] = 1

            trajectories = np.array(
                [x[1:] for x in trajectories]
            )

            rel_sim = rel_sim.to_numpy()
            for j in range(rel_sim.shape[0]):
                if j > 0:
                    patterns = trajectories[:, j - 1]
                    pattern_size = self.encoder.output_sds.size
                    n_patterns = patterns.shape[0]
                    dense_patterns = np.zeros((n_patterns, pattern_size), dtype='float32')
                    for k, p in enumerate(patterns):
                        dense_patterns[k, p] = 1
                else:
                    pattern_size = self.encoder.feedforward_sds.size
                    dense_patterns = dense_start_images

                dim = int(np.sqrt(self.som_size))
                som = MiniSom(
                    dim, dim,
                    pattern_size,
                    sigma=self.som_sigma,
                    learning_rate=self.som_learning_rate,
                    random_seed=self.seed
                )
                som.pca_weights_init(dense_patterns)
                som.train(dense_patterns, self.som_iterations)

                activation_map = np.zeros((dim, dim, 10))
                fig = plt.figure(figsize=(8, 8))
                plt.imshow(som.distance_map(), cmap='Greys', alpha=0.5)
                plt.colorbar()

                for p, cls in zip(dense_patterns, start_classes):
                    activation_map[:, :, cls] += som.activate(p)

                    cell = som.winner(p)
                    plt.text(
                        cell[0],
                        cell[1],
                        str(cls),
                        color=plt.cm.rainbow(cls/10),
                        alpha=0.1,
                        fontdict={'weight': 'bold', 'size': 16}
                    )
                # plt.axis([0, som.get_weights().shape[0], 0, som.get_weights().shape[1]])

                self.logger.log(
                    {
                        'som/clusters': wandb.Image(fig),
                        'iteration': j
                    },
                    step=i
                )
                plt.close('all')

                # normalize activation map
                activation_map /= dense_patterns.shape[0]
                activation_map /= activation_map.sum(axis=-1).reshape((dim, dim, 1))
                # generate colormap
                colors = [plt.cm.rainbow(c/10)[:-1] for c in range(10)]
                color_map = (np.dot(activation_map.reshape((-1, 10)), colors) * 255)
                color_map = color_map.reshape((dim, dim, 3))

                for cls in range(10):
                    self.logger.log(
                        {
                            f'som/activation {cls}': wandb.Image(
                                sns.heatmap(activation_map[:, :, cls], cmap='viridis')
                            )
                        },
                        step=i
                    )
                    plt.close('all')

                self.logger.log(
                    {
                        'som/soft_clusters': wandb.Image(
                            plt.imshow(color_map.astype('uint8'))
                        )
                    },
                    step=i
                )
                plt.close('all')

                out_sim = similarities[:, j]
                hist, x, y = np.histogram2d(in_sim, out_sim)
                x, y = np.meshgrid(x, y)

                self.logger.log(
                    {
                        'main_metrics/relative_similarity': rel_sim[j].mean(),
                        'convergence/io_hist': wandb.Image(
                            plt.pcolormesh(x, y, hist.T)
                        )
                    },
                    step=i
                )

                for cls in range(10):
                    self.logger.log(
                        {
                            f'relative_similarity/class {cls}': rel_sim[j, cls]
                        },
                        step=i
                    )

                i += 1

    def attract(self, steps, pattern, learn=False):
        trajectory = list()

        if self.encoder is not None:
            pattern = self.encoder.compute(pattern, learn=False)

        trajectory.append(pattern)
        for step in range(steps):
            pattern = self.attractor.compute(pattern, learn)
            trajectory.append(pattern)

        return trajectory

    def similarity(self, x1, x2):
        return np.count_nonzero(np.isin(x1, x2)) / x2.size

    def preprocess(self, obs):
        thresh = obs.mean()
        obs = np.flatnonzero(obs >= thresh)
        return obs
