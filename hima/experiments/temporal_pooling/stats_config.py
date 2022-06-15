#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.


class StatsMetricsConfig:
    normalization_unbias: str

    prefix_similarity_discount: float
    loss_layer_discount: float

    def __init__(
            self, normalization_unbias: str, prefix_similarity_discount: float,
            loss_layer_discount: float
    ):
        self.normalization_unbias = normalization_unbias
        self.prefix_similarity_discount = prefix_similarity_discount
        self.loss_layer_discount = loss_layer_discount
