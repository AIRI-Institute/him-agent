#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from hima.common.sdr import SparseSdr
from hima.common.sdrr import RateSdr
from hima.common.sds import Sds


@dataclass
class SdrDataset:
    binary: bool

    sdrs: list[RateSdr]
    _targets: npt.NDArray[float]

    def __init__(self, sdrs, targets, binary: bool):
        self.binary = binary
        self.sdrs = sdrs
        self._targets = targets

    @property
    def n_images(self):
        return len(self.sdrs)

    @property
    def targets(self):
        return self._targets[:, 1:]

    @property
    def target_size(self):
        return self.targets.shape[1]

    def get_sdr(self, ind: int) -> SparseSdr | RateSdr:
        return self.sdrs[ind].sdr if self.binary else self.sdrs[ind]

    def normalize(self, normalizer):
        self.sdrs = [
            RateSdr(sdr=sdr.sdr, values=normalizer(sdr.values))
            for sdr in self.sdrs
        ]


@dataclass
class DvsSdrs:
    n_elements: int
    sds: Sds
    rate_sds: Sds

    sdrs: npt.NDArray[int]
    rates: npt.NDArray[float]
    indices: npt.NDArray[int]

    imu_events: npt.NDArray[float] | None

    def __init__(self, n_elements, sds, rate_sds, sdrs, rates, indices, imu_events=None):
        self.n_elements = n_elements
        self.sds = Sds.make(sds)
        self.rate_sds = Sds.make(rate_sds)
        self.sdrs = sdrs
        self.rates = rates
        self.indices = indices
        self.imu_events = imu_events


class DvsDataset:
    dataset: DvsSdrs

    train: SdrDataset
    test: SdrDataset

    sds: Sds
    binary: bool

    def __init__(self, seed: int, filepath: str | Path, binary: bool = True, normalizer=None):
        self.dataset = _read_dataset(filepath, with_imu=True)
        self.binary = binary
        self.sds = self.dataset.sds if self.binary else self.dataset.rate_sds

        rate_sdrs = unflatten_sdrs(self.dataset.sdrs, self.dataset.rates, self.dataset.indices)
        split_ix = int(len(rate_sdrs) * 0.8)
        train_sdrs = rate_sdrs[:split_ix]
        test_sdrs = rate_sdrs[split_ix:]
        train_targets = self.dataset.imu_events[:split_ix]
        test_targets = self.dataset.imu_events[split_ix:]

        self.train = SdrDataset(train_sdrs, train_targets, binary)
        self.test = SdrDataset(test_sdrs, test_targets, binary)

    @property
    def n_classes(self):
        return self.train.target_size


def _read_dataset(filepath: str, with_imu: bool) -> DvsSdrs:
    filepath = Path(filepath)
    filepath = filepath.expanduser()
    ds_dir = filepath if filepath.is_dir() else filepath.parent

    with open(ds_dir / 'sdrs_info.pkl', mode='rb') as f:
        sdrs = pickle.load(f)

    sdrs |= np.load(ds_dir / 'sdrs.npz')
    if with_imu:
        sdrs |= np.load(ds_dir / 'imu.npz')

    return DvsSdrs(**sdrs)


def unflatten_sdrs(
        sdrs: npt.NDArray[int], rates: npt.NDArray[float],
        indices: npt.NDArray[int]
) -> list[RateSdr]:
    return [
        RateSdr(
            sdrs[indices[i-1]:indices[i]],
            rates[indices[i-1]:indices[i]]
        )
        for i in range(len(indices[1:]))
    ]

