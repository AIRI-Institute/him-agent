#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.
import numpy as np

from hima.common.utils import clip


def generate_synthetic_sequences(
        seed: int,
        n_sequences: int,
        sequence_length: int, alphabet_size: int,
        sequence_similarity: float,
        sequence_similarity_std: float = 0.0
) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # init all sequences from the same origin
    origin_sequence = rng.integers(0, high=alphabet_size, size=(1, sequence_length))
    sequences = origin_sequence.repeat(n_sequences, axis=0)

    # pin 0-th sequence to be the origin
    origin_sequence = sequences[0]

    def _get_target_similarity():
        sim = sequence_similarity
        # if sim std is set, sample individual target similarity for each sequence
        if sequence_similarity_std > 0:
            sim = rng.normal(sequence_similarity, scale=sequence_similarity_std)
            sim = clip(sim, 0, 1)
        return sim

    # perturb each other sequence to have target similarity with the origin
    for i in range(1, n_sequences):
        target_similarity = _get_target_similarity()

        n_elements_to_change = int(sequence_length * (1 - target_similarity))
        if not n_elements_to_change:
            continue

        changing_indices = rng.choice(sequence_length, n_elements_to_change, replace=False)

        # re-sample elements from reduced alphabet space: alphabet_size - 1,
        # because for each element position we exclude ane element value — the one from the origin
        new_values = rng.integers(0, alphabet_size - 1, n_elements_to_change)

        # that's how we exclude one origin value: |0|1|2| -> |0|.|2|3| — value 1 is excluded
        # this means we have to increment values from the tail, 1 and 2 in the example above
        mask = new_values >= origin_sequence[changing_indices]
        new_values[mask] += 1

        # for selected indices, replace origin values with the new different values
        sequences[i, changing_indices] = new_values

    return sequences


def generate_synthetic_single_element_sequences(
        n_sequences: int, alphabet_size: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, high=alphabet_size, size=(n_sequences, 1))
