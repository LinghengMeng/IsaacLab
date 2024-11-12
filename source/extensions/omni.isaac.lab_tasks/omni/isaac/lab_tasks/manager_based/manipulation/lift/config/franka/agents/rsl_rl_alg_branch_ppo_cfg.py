# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from rsl_rl.algorithms import PPO

##############################################
@configclass
class LiftCubePPORunnerCfg():
    seed: int = 42
    # num_steps_per_env = 32
    # max_iterations = 1000
    # save_interval = 50
    experiment_name = "Isaac-Lift-Cube-Franka-v0"
    run_name = ""
    device = "cuda:0"

    resume = False

    alg_class = PPO

    agent_kwargs = dict(
        actor_activations=["elu", "elu", "elu", "linear"],
        actor_hidden_dims=[256, 128, 64],
        actor_input_normalization=True,
        actor_noise_std=1.0,
        batch_count=12,
        clip_ratio=0.2,
        critic_activations=["elu", "elu", "elu", "linear"],
        critic_hidden_dims=[256, 128, 64],
        critic_input_normalization=True,
        entropy_coeff=0.006,
        gae_lambda=0.95,
        gamma=0.98,
        gradient_clip=1.0,
        learning_rate=1.0e-4,
        schedule="adaptive",
        target_kl=0.01,
        value_coeff=1.0,
    )
    runner_kwargs = dict(
        num_steps_per_env=24
    )