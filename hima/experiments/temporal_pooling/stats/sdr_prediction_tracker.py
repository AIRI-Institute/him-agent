#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from __future__ import annotations

import numpy as np

from hima.common.scheduler import Scheduler
from hima.common.sdr import AnySparseSdr, unwrap_as_rate_sdr
from hima.common.sds import Sds
from hima.common.utils import safe_divide
from hima.experiments.temporal_pooling.stats.mean_value import MeanValue, LearningRateParam
from hima.experiments.temporal_pooling.stats.metrics import TMetrics, sdr_similarity


class SdrPredictionTracker:
    sds: Sds
    step_flush_schedule: int | None
    scheduler: Scheduler

    miss_rate: MeanValue[float]
    imprecision: MeanValue[float]
    prediction_volume: MeanValue[float]
    dissimilarity: MeanValue[float]

    def __init__(self, sds: Sds, step_flush_schedule: int = None):
        self.sds = sds
        self.scheduler = Scheduler(step_flush_schedule)

        self.dense_cache = np.zeros(sds.size, dtype=float)
        self.predicted_sdr = []
        self.observed_sdr = []

        lr = LearningRateParam(window=1_000)
        self.miss_rate = MeanValue(lr=lr)
        self.imprecision = MeanValue(lr=lr)
        self.prediction_volume = MeanValue(lr=lr)
        self.dissimilarity = MeanValue(lr=lr)

    def on_sdr_predicted(self, sdr: AnySparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        self.predicted_sdr = sdr
        return {}

    def on_sdr_observed(self, sdr: AnySparseSdr, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        self.observed_sdr = sdr
        return {}

    def on_both_known(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        pr_sdr, pr_value = unwrap_as_rate_sdr(self.predicted_sdr)
        gt_sdr, gt_value = unwrap_as_rate_sdr(self.observed_sdr)

        pr_set_sdr, gt_set_sdr = set(pr_sdr), set(gt_sdr)

        # NB: recall/miss_rate and precision/imprecision are BINARY metrics
        recall = sdr_similarity(
            pr_set_sdr, gt_set_sdr,
            symmetrical=False, dense_cache=self.dense_cache
        )
        miss_rate = 1 - recall
        self.miss_rate.put(miss_rate)

        precision = sdr_similarity(
            gt_set_sdr, pr_set_sdr,
            symmetrical=False, dense_cache=self.dense_cache
        )
        imprecision = 1 - precision
        self.imprecision.put(imprecision)

        prediction_volume = safe_divide(len(pr_sdr), self.sds.active_size)
        self.prediction_volume.put(prediction_volume)

        # NB: ...while similarity/dissimilarity is RATE metric [if sdrs are rate sdrs]
        similarity = sdr_similarity(
            self.predicted_sdr, self.observed_sdr,
            symmetrical=True, dense_cache=self.dense_cache
        )
        dissimilarity = 1 - similarity
        self.dissimilarity.put(dissimilarity)

        self.predicted_sdr = None
        self.observed_sdr = None

        if self.scheduler.tick():
            return self.flush_step_metrics()
        return {}

    def on_sequence_finished(self, _, ignore: bool) -> TMetrics:
        if ignore:
            return {}

        if self.scheduler.is_infinite:
            return self.flush_step_metrics()
        return {}

    def flush_step_metrics(self) -> TMetrics:
        miss_rate = self.miss_rate.get()
        imprecision = self.imprecision.get()
        f1_score = safe_divide(
            2 * (1 - miss_rate) * (1 - imprecision),
            (1 - miss_rate) + (1 - imprecision)
        )

        metrics = {
            'f1_score': f1_score,
            'miss_rate': miss_rate,
            'dissimilarity': self.dissimilarity.get(),
            'imprecision': imprecision,
            'prediction_volume': self.prediction_volume.get(),
        }
        self._reset_step_metrics()
        return metrics

    def _reset_step_metrics(self):
        pass


def get_sdr_prediction_tracker(on: dict, **config) -> SdrPredictionTracker:
    gt_stream = on['sdr_observed']
    return SdrPredictionTracker(sds=gt_stream.sds, **config)
