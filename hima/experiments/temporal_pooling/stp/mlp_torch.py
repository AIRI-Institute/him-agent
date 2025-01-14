#  Copyright (c) 2023 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import nn, optim

from hima.experiments.temporal_pooling.stp.mlp_decoder_torch import SymExpModule


class MlpClassifier:
    type: str
    is_classifier: bool
    layer_dims: list[int]

    _lr_epoch_step: int
    _lr_epoch_steps: int

    def __init__(
            self,
            classification: bool, layers: list[int],
            learning_rate: float,
            seed: int = None,
            collect_losses: int = 0,
            n_extra_layers: int | list[int] = 0,
            symexp_logits: bool = False,
    ):
        self.rng = np.random.default_rng(seed)
        self.lr = learning_rate

        self.is_classifier = classification
        if self.is_classifier:
            self.type = 'CE classifier'
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.type = 'MSE regressor'
            self.loss_func = nn.MSELoss()

        if isinstance(n_extra_layers, list):
            out_size = layers.pop()
            for layer in n_extra_layers:
                layers.append(layer)
            layers.append(out_size)
            print(f'MLP {self.type} layers: {layers}')
        elif n_extra_layers > 0:
            out_size = layers.pop()
            for _ in range(n_extra_layers):
                layers.append(layers[-1] // 2)
            layers.append(out_size)
            print(f'MLP {self.type} layers: {layers}')

        self.layer_dims = layers

        nn_layers = [
            nn.Linear(layers[0], layers[1], dtype=float)
        ]
        for i in range(1, len(layers) - 1):
            nn_layers.append(nn.SiLU())
            nn_layers.append(nn.Linear(layers[i], layers[i + 1], dtype=float))

        if symexp_logits:
            nn_layers.append(SymExpModule())
        self.mlp = nn.Sequential(*nn_layers)

        self.optim = optim.Adam(self.mlp.parameters(), lr=self.lr, weight_decay=1e-6)
        # self.optim = optim.Adam(self.mlp.parameters(), lr=self.lr)

        self._min_lr = self.lr / 20.0
        self._lr_epoch = lambda lr: int(7.0 / lr)
        self._lr_scaler = lambda _: 0.8
        self.lr_scheduler = optim.lr_scheduler.MultiplicativeLR(
            self.optim, lr_lambda=self._lr_scaler,
        )
        self._lr_epoch_step = 0
        self._lr_epoch_steps = self._lr_epoch(self.lr)

        self.collect_losses = collect_losses
        if self.collect_losses > 0:
            from collections import deque
            self.losses = deque(maxlen=self.collect_losses)

    def predict(self, dense_sdr: npt.NDArray[float]) -> npt.NDArray[float]:
        dense_sdr = torch.from_numpy(dense_sdr)
        with torch.no_grad():
            return self.mlp(dense_sdr).numpy()

    def learn(self, batch_dense_sdr: npt.NDArray[float], targets: npt.NDArray[int]):
        batch_dense_sdr = torch.from_numpy(batch_dense_sdr)
        targets = torch.from_numpy(targets)

        self.optim.zero_grad()

        loss = self.loss_func(self.mlp(batch_dense_sdr), targets)
        loss.backward()
        self.optim.step()

        if self.collect_losses:
            self.losses.append(loss.item())

        self._lr_epoch_step += 1
        if self.lr > self._min_lr and self._lr_epoch_step >= self._lr_epoch_steps:
            self.lr *= self._lr_scaler(True)
            self._lr_epoch_step = 0
            self._lr_epoch_steps = self._lr_epoch(self.lr)
            # print(f'New LR: {self.lr:.5f} for {self._lr_epoch_steps} steps')
            self.lr_scheduler.step()

    @property
    def input_size(self):
        return self.layer_dims[0]

    @property
    def output_size(self):
        return self.layer_dims[-1]
