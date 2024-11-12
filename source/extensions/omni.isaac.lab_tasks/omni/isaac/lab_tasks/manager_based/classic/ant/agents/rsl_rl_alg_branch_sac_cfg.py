# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass
from rsl_rl.algorithms import PPO, TD3, SAC

##############################################
@configclass
class AntSACRunnerCfg():
    seed: int = 42
    # num_steps_per_env = 32
    # max_iterations = 1000
    # save_interval = 50
    experiment_name = "ant_sac"
    run_name = ""
    device = "cuda:0"

    resume = False

    alg_class = SAC

    agent_kwargs = dict(
        actor_activations=["relu", "relu", "tanh"],
        actor_hidden_dims=[256, 256],
        actor_input_normalization=True,    
        action_max = 1,
        action_min = -1,
        batch_count=1,
        batch_size=100,
        critic_activations=["relu", "relu", "linear"],
        critic_hidden_dims=[256, 256],
        critic_input_normalization=True,
        polyak = 0.995,
        actor_lr = 1e-4,
        critic_lr = 1e-3,
        storage_initial_size = 0, 
        storage_size = 1000000
    )
    runner_kwargs = dict(
        num_steps_per_env=24
    )