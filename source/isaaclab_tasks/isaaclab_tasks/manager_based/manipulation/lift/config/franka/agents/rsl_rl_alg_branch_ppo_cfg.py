# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass
from rsl_rl.algorithms import PPO

##############################################
@configclass
class LiftCubePPORunnerCfg():
    """alg_branch (LinghengMeng/rsl_rl fork) translation of this task's own
    rsl_rl_ppo_cfg.py:LiftCubePPORunnerCfg (standard isaaclab_rl.rsl_rl API) —
    same values, just in agent_kwargs/runner_kwargs dict form. num_learning_epochs=5
    x num_mini_batches=4 has no direct equivalent here (our fork's PPO.update()
    does exactly `batch_count` minibatch steps in a single pass per rollout, no
    outer epoch loop — see PPL_Implementation_Plan.md's 2026-08-14 Ant
    hyperparameter-gap section) — batch_count=20 approximates the same total
    gradient-step count per rollout.
    """
    seed: int = 42
    # num_steps_per_env = 24
    # max_iterations = 1500
    save_interval = 50          # needed for saving checkpoint
    experiment_name = "franka_lift"
    run_name = "franka_lift"
    device = "cuda:0"

    resume = False

    alg_class = PPO

    agent_kwargs = dict(
        actor_activations=["elu", "elu", "elu", "linear"],
        actor_hidden_dims=[256, 128, 64],
        actor_input_normalization=True,
        actor_noise_std=1.0,
        batch_count=20,
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
