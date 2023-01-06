#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
from hima.modules.htm.spatial_pooler import SPDecoder, HtmSpatialPooler
from htm.bindings.sdr import SDR
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == '__main__':
    params = dict(
        potentialPct=0.5,
        globalInhibition=True,
        localAreaDensity=0,
        numActiveColumnsPerInhArea=1,
        stimulusThreshold=1,
        synPermInactiveDec=0.01,
        synPermActiveInc=0.1,
        synPermConnected=0.5,
        minPctOverlapDutyCycle=0.001,
        dutyCyclePeriod=1000,
        boostStrength=0.0,
        seed=432,
        spVerbosity=0,
        wrapAround=False
    )

    sp = HtmSpatialPooler([5, 5], [3, 3], **params)
    decoder = SPDecoder(sp)

    probs = np.random.rand(sp.getNumColumns())

    res = decoder.decode(probs, update=True)

    sns.heatmap(res.reshape(sp.getInputDimensions()))
    plt.show()