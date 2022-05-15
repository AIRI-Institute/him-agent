#  Copyright (c) 2022 Autonomous Non-Profit Organization "Artificial Intelligence Research
#  Institute" (AIRI); Moscow Institute of Physics and Technology (National Research University).
#  All rights reserved.
#
#  Licensed under the AGPLv3 license. See LICENSE in the project root for license information.

from hima.common.config_utils import extracted_type, resolve_init_params, extracted
from hima.common.utils import ensure_absolute_number
from hima.modules.htm.temporal_memory import ClassicApicalTemporalMemory, DelayedFeedbackTM


def resolve_tp(config, temporal_pooler: str, temporal_memory):
    base_config_tp = config['temporal_poolers'][temporal_pooler]
    seed = config['seed']
    input_size = temporal_memory.columns * temporal_memory.cells_per_column if not \
        isinstance(temporal_memory, ClassicApicalTemporalMemory)  \
        else temporal_memory.columns

    config_tp = dict(
        inputDimensions=[input_size],
        potentialRadius=input_size,
    )

    base_config_tp, tp_type = extracted_type(base_config_tp)
    if tp_type == 'UnionTp':
        from hima.modules.htm.spatial_pooler import UnionTemporalPooler
        config_tp = base_config_tp | config_tp
        tp = UnionTemporalPooler(seed=seed, **config_tp)
    elif tp_type == 'AblationUtp':
        from hima.experiments.temporal_pooling.ablation_utp import AblationUtp
        config_tp = base_config_tp | config_tp
        tp = AblationUtp(seed=seed, **config_tp)
    elif tp_type == 'CustomUtp':
        from hima.experiments.temporal_pooling.custom_utp import CustomUtp
        config_tp = base_config_tp | config_tp
        del config_tp['potentialRadius']
        tp = CustomUtp(seed=seed, **config_tp)
    elif tp_type == 'SandwichTp':
        # FIXME: dangerous mutations here! We should work with copies
        from hima.experiments.temporal_pooling.sandwich_tp import SandwichTp
        base_config_tp['lower_sp_conf'] = base_config_tp['lower_sp_conf'] | config_tp
        base_config_tp['lower_sp_conf']['seed'] = seed
        base_config_tp['upper_sp_conf']['seed'] = seed
        tp = SandwichTp(**base_config_tp)
    else:
        raise KeyError(f'Temporal Pooler type "{tp_type}" is not supported')
    return tp


def resolve_tm(config, action_encoder, state_encoder):
    return make_context_tm(config, action_encoder, state_encoder)


def make_context_tm(config, action_encoder, state_encoder):
    base_config_tm = config['temporal_memory']
    seed = config['seed']

    # apical feedback
    apical_feedback_cells = base_config_tm['feedback_cells']
    apical_active_bits = ensure_absolute_number(
        base_config_tm['sample_size_apical'],
        baseline=apical_feedback_cells
    )
    activation_threshold_apical = ensure_absolute_number(
        base_config_tm['activation_threshold_apical'],
        baseline=apical_active_bits
    )
    learning_threshold_apical = ensure_absolute_number(
        base_config_tm['learning_threshold_apical'],
        baseline=apical_active_bits
    )

    # basal context
    basal_active_bits = state_encoder.n_active_bits

    config_tm = dict(
        columns=action_encoder.output_sdr_size,

        feedback_cells=apical_feedback_cells,
        sample_size_apical=apical_active_bits,
        activation_threshold_apical=activation_threshold_apical,
        learning_threshold_apical=learning_threshold_apical,
        max_synapses_per_segment_apical=apical_active_bits,

        context_cells=state_encoder.output_sdr_size,
        sample_size_basal=basal_active_bits,
        activation_threshold_basal=basal_active_bits,
        learning_threshold_basal=basal_active_bits,
        max_synapses_per_segment_basal=basal_active_bits,
    )

    # it's necessary as we shadow some "relative" values with the "absolute" values
    config_tm = base_config_tm | config_tm
    tm = DelayedFeedbackTM(seed=seed, **config_tm)
    return tm


def resolve_run_setup(config: dict, run_setup_config):
    if isinstance(run_setup_config, str):
        run_setup_config = config['run_setups'][run_setup_config]

    from hima.experiments.temporal_pooling.new.test_on_policies import RunSetup
    return RunSetup(**run_setup_config)


def resolve_data_generator(config: dict, **induction_registry):
    generator_config, generator_type = extracted_type(config['generator'])

    if generator_type == 'synthetic':
        from hima.experiments.temporal_pooling.data_generation import SyntheticGenerator
        generator_config = resolve_init_params(generator_config, **induction_registry)
        return SyntheticGenerator(config, **generator_config)
    elif generator_type == 'aai_rotation':
        from hima.experiments.temporal_pooling.data_generation import AAIRotationsGenerator
        return AAIRotationsGenerator(config)
    else:
        raise KeyError(f'{generator_type} is not supported')


def resolve_encoder(
        config: dict, key, registry_key: str,
        n_values: int = None, active_size: int = None, seed: int = None
):
    registry = config[registry_key]
    encoder_config, encoder_type = extracted_type(registry[key])

    if encoder_type == 'int_bucket':
        from hima.common.sdr_encoders import IntBucketEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, bucket_size=active_size
        )
        return IntBucketEncoder(**encoder_config)
    if encoder_type == 'int_random':
        from hima.common.sdr_encoders import IntRandomEncoder
        encoder_config = resolve_init_params(
            encoder_config, n_values=n_values, active_size=active_size, seed=seed
        )
        return IntRandomEncoder(**encoder_config)
    else:
        raise KeyError(f'{encoder_type} is not supported')
