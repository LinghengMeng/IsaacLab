# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class AntPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32
    max_iterations = 1000
    save_interval = 50
    experiment_name = "ant"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

##############################################

agent_kwargs = dict(
        actor_activations=["elu", "elu", "elu", "linear"],
        actor_hidden_dims=[400, 200, 100],
        actor_input_normalization=False,
        actor_noise_std=0.2611,
        batch_count=12,
        clip_ratio=0.2,
        critic_activations=["elu", "elu", "elu", "linear"],
        critic_hidden_dims=[400, 200, 100],
        critic_input_normalization=False,
        entropy_coeff=0.0,
        gae_lambda=0.95,
        gamma=0.99,
        gradient_clip=5.0,
        learning_rate=0.8755,
        schedule="adaptive",
        target_kl=0.01,
        value_coeff=1.0,
    )
runner_kwargs = dict(
        num_steps_per_env=32
    )