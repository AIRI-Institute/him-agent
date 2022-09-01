#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from typing import Any

import numpy as np
from htm.bindings.algorithms import SpatialPooler
from htm.bindings.sdr import SDR

from hima.common.config_utils import resolve_init_params, extracted
from hima.common.sdr import SparseSdr
from hima.common.sds import Sds
from hima.experiments.temporal_pooling.new.blocks.graph import Block
from hima.experiments.temporal_pooling.new.blocks.base_block_stats import BlockStats
from hima.experiments.temporal_pooling.new.sdr_seq_stats import SdrSequenceStats
from hima.experiments.temporal_pooling.new.stats_config import StatsMetricsConfig


class SpatialPoolerBlockStats(BlockStats):
    seq_stats: SdrSequenceStats

    def __init__(self, output_sds: Sds):
        super(SpatialPoolerBlockStats, self).__init__(output_sds)
        self.seq_stats = SdrSequenceStats(self.output_sds)

    def update(self, current_output_sdr: SparseSdr):
        self.seq_stats.update(current_output_sdr)

    def step_metrics(self) -> dict[str, Any]:
        return self.seq_stats.step_metrics()

    def final_metrics(self) -> dict[str, Any]:
        return self.seq_stats.final_metrics()


class SpatialPoolerBlock(Block):
    family = "spatial_pooler"

    output_sdr: SparseSdr
    sp: Any

    _active_input: SDR
    _active_output: SDR

    def __init__(self, id_: int, name: str, **sp_config):
        super(SpatialPoolerBlock, self).__init__(id_, name)

        sp_config, ff_sds, output_sds = extracted(sp_config, 'ff_sds', 'output_sds')

        self.sds = {}
        self.resolve_sds('feedforward', ff_sds)
        self.resolve_sds('output', output_sds)

        self.output_sdr = []
        self._sp_config = sp_config

    def build(self):
        sp_config = self._sp_config
        ff_sds = self.feedforward_sds
        output_sds = self.output_sds

        # if FF/Out SDS was defined in config, they aren't Sds objects, hence explicit conversion
        ff_sds = Sds.as_sds(ff_sds)
        output_sds = Sds.as_sds(output_sds)

        sp_config = resolve_init_params(
            sp_config,
            inputDimensions=ff_sds.shape, potentialRadius=ff_sds.size,
            columnDimensions=output_sds.shape, localAreaDensity=output_sds.sparsity,
        )
        self.sp = SpatialPooler(
            **sp_config
        )
        self._active_input = SDR(self.feedforward_sds.size)
        self._active_output = SDR(self.output_sds.size)

    def track_stats(self, name: str, stats_config: StatsMetricsConfig):
        self.stats[name] = SpatialPoolerBlockStats(self.sds[name])

    def reset_stats(self):
        for name in self.stats:
            self.stats[name] = SpatialPoolerBlockStats(self.sds[name])

    @property
    def tag(self) -> str:
        return f'{self.id}_sp'

    def reset(self):
        self._active_input.sparse = []
        self._active_output.sparse = []

    def compute(self, data: dict[str, SparseSdr], **kwargs):
        self._compute(**data, **kwargs)

    def _compute(self, feedforward: SparseSdr, learn: bool = True) -> SparseSdr:
        self._active_input.sparse = feedforward.copy()

        self.sp.compute(self._active_input, learn=learn, output=self._active_output)
        self.output_sdr = np.array(self._active_output.sparse, copy=True)

        self.stats['output'].update(current_output_sdr=self.output_sdr)
        return self.output_sdr


def resolve_sp(sp_config, block_id: int, block_name: str, **induction_registry):
    sp_config = resolve_init_params(sp_config, raise_if_not_resolved=False, **induction_registry)
    return SpatialPoolerBlock(block_id, block_name, **sp_config)
