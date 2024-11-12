# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from rsl_rl.algorithms import PPO, TD3

##############################################
@configclass
class AntTD3RunnerCfg():
    seed: int = 42
    # num_steps_per_env = 32
    # max_iterations = 1000
    # save_interval = 50
    experiment_name = "ant"
    run_name = ""
    device = "cuda:0"

    resume = False

    alg_class = TD3

    agent_kwargs = dict(
        actor_activations=["relu", "relu", "tanh"],
        actor_hidden_dims=[256, 256],
        actor_input_normalization=True,    
        action_noise_scale = 0.1, # std of the Gaussian actio noise
        action_max = 1,
        action_min = -1,
        batch_count=1,
        batch_size=100,
        critic_activations=["relu", "relu", "linear"],
        critic_hidden_dims=[256, 256],
        critic_input_normalization=True,
        polyak = 0.995,
        actor_lr = 1e-3,
        critic_lr = 1e-3,
        noise_clip = 0.5,       # The clipped noise range [-noise_clip, noise_clip]
        policy_delay = 2,
        target_noise_scale = 0.2,
        storage_initial_size = 0, 
        storage_size = 1000000
    )
    runner_kwargs = dict(
        num_steps_per_env=24
    )