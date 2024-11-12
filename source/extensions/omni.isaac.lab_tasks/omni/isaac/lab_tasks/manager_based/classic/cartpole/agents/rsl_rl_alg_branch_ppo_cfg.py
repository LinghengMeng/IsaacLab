# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from rsl_rl.algorithms import PPO

##############################################
@configclass
class CartpolePPORunnerCfg():
    seed: int = 42
    # num_steps_per_env = 32
    # max_iterations = 1000
    # save_interval = 50
    experiment_name = "cartpole"
    run_name = ""
    device = "cuda:0"

    resume = False

    alg_class = PPO

    agent_kwargs = dict(
        actor_activations=["elu", "elu", "linear"],  # checked
        actor_hidden_dims=[32, 32],                  # checked
        actor_input_normalization=True,
        actor_noise_std=1.0,
        # batch_size = total_size // batch_count
        # total_size = num_envs * num_steps_per_env
        batch_count=4,    # 12 not work (if set to 20, cause NAN, but if num_steps_per_env > 20 there is no NAN.)
        clip_ratio=0.2,                              # checked
        critic_activations=["elu", "elu", "linear"], # checked
        critic_hidden_dims=[32, 32],                 # checked
        critic_input_normalization=True,
        entropy_coeff=0.005,                         # checked
        gae_lambda=0.95,                             # checked
        gamma=0.99,                                  # checked     
        gradient_clip=1.0,                           # checked
        learning_rate=1.0e-3,                        # checked
        schedule="adaptive",                         # checked
        target_kl=0.01,                              # checked
        value_coeff=1.0,                             # checked
    )
    runner_kwargs = dict(
        num_steps_per_env=16                         # checked
    )

