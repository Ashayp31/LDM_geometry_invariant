from enum import Enum
from logging import Logger
from typing import List, Any

import numpy as np

from src_vqvae.handlers.general import ParamSchedulerHandler
from src_vqvae.networks.vqvae.baseline import BaselineVQVAE
from src_vqvae.networks.vqvae.single_vqvae import SingleVQVAE
from src_vqvae.networks.vqvae.vqvae import VQVAEBase
from src_vqvae.utils.general import get_max_decay_epochs
from src_vqvae.utils.constants import DecayWarmups


class VQVAENetworks(Enum):
    SINGLE_VQVAE = "single_vqvae"
    BASELINE_VQVAE = "baseline_vqvae"
    SLIM_VQVAE = "slim_vqvae"


def get_vqvae_network(config, conditioning_network=False) -> VQVAEBase:

    config = vars(config)
    config_name_add = "" if not conditioning_network else "_conditioning"
    network = BaselineVQVAE(
        n_levels=config["no_levels" + config_name_add],
        downsample_parameters=config["downsample_parameters" + config_name_add],
        upsample_parameters=config["upsample_parameters" + config_name_add],
        n_embed=config["num_embeddings" + config_name_add][0],
        embed_dim=config["embedding_dim" + config_name_add][0],
        commitment_cost=config["commitment_cost" + config_name_add][0],
        n_channels=config["no_channels" + config_name_add],
        n_res_channels=config["no_channels" + config_name_add],
        n_res_layers=config["no_res_layers" + config_name_add],
        dropout_at_end=config["dropout_penultimate" + config_name_add],
        dropout_enc=config["dropout_enc" + config_name_add],
        dropout_dec=config["dropout_dec" + config_name_add],
        output_act=config["output_act" + config_name_add],
        vq_decay=config["decay" + config_name_add][0],
        use_subpixel_conv=config["use_subpixel_conv" + config_name_add],
        apply_coordConv=config["apply_coordConv" + config_name_add],
    )

    return network


def add_vqvae_network_handlers(
    train_handlers: List, vqvae: VQVAEBase, config: Any, logger: Logger
) -> List:

    if config.decay_warmup == DecayWarmups.STEP.value:
        delta_step = (0.99 - config.decay[0]) / 4
        stair_steps = np.linspace(0, config.max_decay_epochs, 5)[1:]

        def decay_anealing(current_step: int) -> float:
            if (current_step + 1) >= 200:
                return 0.99
            if (current_step + 1) >= 160:
                return 0.95
            if (current_step + 1) >= 80:
                return 0.80
            if (current_step + 1) >= 40:
                return 0.70
            return 0.5

        #def decay_anealing(current_step: int) -> float:
        #    if (current_step + 1) >= stair_steps[3]:
        #        return config.decay[0] + 4 * delta_step
        #    if (current_step + 1) >= stair_steps[2]:
        #        return config.decay[0] + 3 * delta_step
        #    if (current_step + 1) >= stair_steps[1]:
        #        return config.decay[0] + 2 * delta_step
        #    if (current_step + 1) >= stair_steps[0]:
        #        return config.decay[0] + delta_step
        #    return config.decay[0]

        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator=decay_anealing,
                vc_kwargs={},
                epoch_level=True,
            )
        ]
    elif config.decay_warmup == DecayWarmups.LINEAR.value:
        train_handlers += [
            ParamSchedulerHandler(
                parameter_setter=vqvae.set_ema_decay,
                value_calculator="linear",
                vc_kwargs={
                    "initial_value": config.decay,
                    "step_constant": 0,
                    "step_max_value": config.max_decay_epochs
                    if isinstance(config.max_decay_epochs, int)
                    else get_max_decay_epochs(config=config, logger=logger),
                    "max_value": 0.99,
                },
                epoch_level=True,
            )
        ]

    return train_handlers

