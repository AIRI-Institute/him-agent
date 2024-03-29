#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.experiments.temporal_pooling.graph.block import Block


class StorageBlock(Block):
    family = "storage"

    def compile(self):
        pass
