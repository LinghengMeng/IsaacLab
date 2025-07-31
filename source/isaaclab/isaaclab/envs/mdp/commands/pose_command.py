# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for pose tracking."""

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import combine_frame_transforms, compute_pose_error, quat_from_euler_xyz, quat_unique

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

    from .commands_cfg import UniformPoseCommandCfg


class UniformPoseCommand(CommandTerm):
    """Command generator for generating pose commands uniformly.

    The command generator generates poses by sampling positions uniformly within specified
    regions in cartesian space. For orientation, it samples uniformly the euler angles
    (roll-pitch-yaw) and converts them into quaternion representation (w, x, y, z).

    The position and orientation commands are generated in the base frame of the robot, and not the
    simulation world frame. This means that users need to handle the transformation from the
    base frame to the simulation world frame themselves.

    .. caution::

        Sampling orientations uniformly is not strictly the same as sampling euler angles uniformly.
        This is because rotations are defined by 3D non-Euclidean space, and the mapping
        from euler angles to rotations is not one-to-one.

    """

    cfg: UniformPoseCommandCfg
    """Configuration for the command generator."""

    def __init__(self, cfg: UniformPoseCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator class.

        Args:
            cfg: The configuration parameters for the command generator.
            env: The environment object.
        """
        # initialize the base class
        super().__init__(cfg, env)

        # extract the robot and body index for which the command is generated
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.body_idx = self.robot.find_bodies(cfg.body_name)[0][0]

        # create buffers
        # -- commands: (x, y, z, qw, qx, qy, qz) in root frame
        self.pose_command_b = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_b[:, 3] = 1.0
        self.pose_command_w = torch.zeros_like(self.pose_command_b)
        # -- metrics
        self.metrics["position_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["orientation_error"] = torch.zeros(self.num_envs, device=self.device)

        # side length of the key points (note: the side length must be euqal)
        self.key_point_side_length = 0.1 # 0.3
        self.pose_command_t_key_point_side_length = self.key_point_side_length
        self.ee_pose_key_point_side_length = self.key_point_side_length
        # add 3 key points relative to the target pose frame
        #    key point 0: (0, 0, 0)
        self.pose_command_t_key_point_0 = torch.zeros(self.num_envs, 7, device=self.device)    # (0,0,0)
        self.pose_command_t_key_point_0[:, 3] = 1.0  
        #    key point 1: (side_length, 0, 0)
        self.pose_command_t_key_point_1 = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_t_key_point_1[:, 0] = self.pose_command_t_key_point_side_length      # (self.ose_command_t_key_point_side_length, 0, 0)
        self.pose_command_t_key_point_1[:, 3] = 1.0     
        #    key point 2: (0, side_length, 0)                                       # quaternion (1.0, 0, 0, 0)
        self.pose_command_t_key_point_2 = torch.zeros(self.num_envs, 7, device=self.device)
        self.pose_command_t_key_point_2[:, 1] = self.pose_command_t_key_point_side_length      # (0, self.ose_command_t_key_point_side_length, 0)
        self.pose_command_t_key_point_2[:, 3] = 1.0 
        # key points in base frame
        self.pose_command_b_key_point_0 = torch.zeros_like(self.pose_command_b)
        self.pose_command_b_key_point_1 = torch.zeros_like(self.pose_command_b)
        self.pose_command_b_key_point_2 = torch.zeros_like(self.pose_command_b)
        # key points in world frame                                           
        self.pose_command_w_key_point_0 = torch.zeros_like(self.pose_command_b)
        self.pose_command_w_key_point_1 = torch.zeros_like(self.pose_command_b)
        self.pose_command_w_key_point_2 = torch.zeros_like(self.pose_command_b)

        # add 3 key points relative to the end-effector pose frame
        #    key point 0: (0, 0, 0)
        self.ee_pose_key_point_0 = torch.zeros(self.num_envs, 7, device=self.device)    # (0,0,0)
        self.ee_pose_key_point_0[:, 3] = 1.0    
        #    key point 1: (side_length, 0, 0)                                        # quaternion (1.0, 0, 0, 0) 
        self.ee_pose_key_point_1 = torch.zeros(self.num_envs, 7, device=self.device)
        self.ee_pose_key_point_1[:, 0] = self.ee_pose_key_point_side_length      # (self.ee_pose_key_point_side_length, 0, 0)
        self.ee_pose_key_point_1[:, 3] = 1.0   
        #    key point 2: (0, side_length, 0)                                           # quaternion (1.0, 0, 0, 0)
        self.ee_pose_key_point_2 = torch.zeros(self.num_envs, 7, device=self.device)
        self.ee_pose_key_point_2[:, 1] = self.ee_pose_key_point_side_length      # (0, self.ee_pose_key_point_side_length, 0)
        self.ee_pose_key_point_2[:, 3] = 1.0                                            # quaternion (1.0, 0,
        self.ee_pose_w_key_point_0 = torch.zeros_like(self.pose_command_b)
        self.ee_pose_w_key_point_1 = torch.zeros_like(self.pose_command_b)
        self.ee_pose_w_key_point_2 = torch.zeros_like(self.pose_command_b)

    def __str__(self) -> str:
        msg = "UniformPoseCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        return msg

    """
    Properties
    """
    @property
    def target_key_points(self) -> torch.Tensor:
        """The target key points in the command. Shape is (num_envs, 21).

        The first three elements correspond to the position of key point 0, followed by the quaternion orientation
        in (w, x, y, z), then the position of key point 1, followed by the quaternion orientation in (w, x, y, z),
        and finally the position of key point 2, followed by the quaternion orientation in (w, x, y, z).
        """
        return torch.cat([self.pose_command_w_key_point_0[:, :3], self.pose_command_w_key_point_1[:, :3], self.pose_command_w_key_point_2[:, :3]], dim=-1)
    
    @property
    def ee_pose_key_points(self) -> torch.Tensor:
        """The end-effector pose key points in the command. Shape is (num_envs, 21).

        The first three elements correspond to the position of key point 0, followed by the quaternion orientation
        in (w, x, y, z), then the position of key point 1, followed by the quaternion orientation in (w, x, y, z),
        and finally the position of key point 2, followed by the quaternion orientation in (w, x, y, z).
        """
        return torch.cat([self.ee_pose_w_key_point_0[:, :3], self.ee_pose_w_key_point_1[:, :3], self.ee_pose_w_key_point_2[:, :3]], dim=-1)

    @property
    def command_key_points(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        # Note use the base frame key points for the command 
        # (Important note: the reason using world frame causing bad performance might be because all other terms in observation are 
        # represented in base frame or using base frame is easier to learn the mapping from joint to target?)
        pose_command_key_point = torch.cat([self.pose_command_b_key_point_0[:, :3], self.pose_command_b_key_point_1[:, :3], self.pose_command_b_key_point_2[:, :3]], dim=-1)
        return pose_command_key_point
        # import pdb; pdb.set_trace()
        # return self.pose_command_b
    
    @property
    def command_key_point_side_length(self) -> float:
        """The side length of the key points in the command."""
        return self.key_point_side_length * torch.ones(self.num_envs, 1, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """The desired pose command. Shape is (num_envs, 7).

        The first three elements correspond to the position, followed by the quaternion orientation in (w, x, y, z).
        """
        return self.pose_command_b

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # transform command from base frame to simulation world frame
        self.pose_command_w[:, :3], self.pose_command_w[:, 3:] = combine_frame_transforms(
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
        )
        # transform key points from target frame to base frame
        self.pose_command_b_key_point_0[:, :3], self.pose_command_b_key_point_0[:, 3:] = combine_frame_transforms(
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
            self.pose_command_t_key_point_0[:, :3],
            self.pose_command_t_key_point_0[:, 3:],
        )
        self.pose_command_b_key_point_1[:, :3], self.pose_command_b_key_point_1[:, 3:] = combine_frame_transforms(
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
            self.pose_command_t_key_point_1[:, :3],
            self.pose_command_t_key_point_1[:, 3:],
        )
        self.pose_command_b_key_point_2[:, :3], self.pose_command_b_key_point_2[:, 3:] = combine_frame_transforms(
            self.pose_command_b[:, :3],
            self.pose_command_b[:, 3:],
            self.pose_command_t_key_point_2[:, :3],
            self.pose_command_t_key_point_2[:, 3:],
        )
        # transform key points from target frame to simulation world frame
        self.pose_command_w_key_point_0[:, :3], self.pose_command_w_key_point_0[:, 3:] = combine_frame_transforms(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.pose_command_t_key_point_0[:, :3],
            self.pose_command_t_key_point_0[:, 3:],
        )
        self.pose_command_w_key_point_1[:, :3], self.pose_command_w_key_point_1[:, 3:] = combine_frame_transforms(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.pose_command_t_key_point_1[:, :3],
            self.pose_command_t_key_point_1[:, 3:],
        )
        self.pose_command_w_key_point_2[:, :3], self.pose_command_w_key_point_2[:, 3:] = combine_frame_transforms(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.pose_command_t_key_point_2[:, :3],
            self.pose_command_t_key_point_2[:, 3:],
        )
        # import pdb; pdb.set_trace()
        # compute the error
        pos_error, rot_error = compute_pose_error(
            self.pose_command_w[:, :3],
            self.pose_command_w[:, 3:],
            self.robot.data.body_pos_w[:, self.body_idx],
            self.robot.data.body_quat_w[:, self.body_idx],
        )
        self.metrics["position_error"] = torch.norm(pos_error, dim=-1)
        self.metrics["orientation_error"] = torch.norm(rot_error, dim=-1)
        # compute the key points error
        #    Convert body frame to world frame
        self.ee_pose_w_key_point_0[:, :3], self.ee_pose_w_key_point_0[:, 3:] = combine_frame_transforms(
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
            self.ee_pose_key_point_0[:, :3],
            self.ee_pose_key_point_0[:, 3:],
        )
        self.ee_pose_w_key_point_1[:, :3], self.ee_pose_w_key_point_1[:, 3:] = combine_frame_transforms(
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
            self.ee_pose_key_point_1[:, :3],
            self.ee_pose_key_point_1[:, 3:],
        )
        self.ee_pose_w_key_point_2[:, :3], self.ee_pose_w_key_point_2[:, 3:] = combine_frame_transforms(
            self.robot.data.body_state_w[:, self.body_idx, :3],
            self.robot.data.body_state_w[:, self.body_idx, 3:7],
            self.ee_pose_key_point_2[:, :3],
            self.ee_pose_key_point_2[:, 3:],
        )
        # compute the key points error
        pos_error_key_point_0, rot_error_key_point_0 = compute_pose_error(
            self.pose_command_t_key_point_0[:, :3], 
            self.pose_command_t_key_point_0[:, 3:], 
            self.ee_pose_w_key_point_0[:, :3], 
            self.ee_pose_w_key_point_0[:, 3:]
        )
        pos_error_key_point_1, rot_error_key_point_1 = compute_pose_error(
            self.pose_command_t_key_point_1[:, :3], 
            self.pose_command_t_key_point_1[:, 3:], 
            self.ee_pose_w_key_point_1[:, :3], 
            self.ee_pose_w_key_point_1[:, 3:]
        )
        pos_error_key_point_2, rot_error_key_point_2 = compute_pose_error(
            self.pose_command_t_key_point_2[:, :3], 
            self.pose_command_t_key_point_2[:, 3:], 
            self.ee_pose_w_key_point_2[:, :3], 
            self.ee_pose_w_key_point_2[:, 3:]
        )
        self.metrics["position_error_key_point_0"] = torch.norm(pos_error_key_point_0, dim=-1)
        self.metrics["orientation_error_key_point_0"] = torch.norm(rot_error_key_point_0, dim=-1)
        self.metrics["position_error_key_point_1"] = torch.norm(pos_error_key_point_1, dim=-1)
        self.metrics["orientation_error_key_point_1"] = torch.norm(rot_error_key_point_1, dim=-1)
        self.metrics["position_error_key_point_2"] = torch.norm(pos_error_key_point_2, dim=-1)
        self.metrics["orientation_error_key_point_2"] = torch.norm(rot_error_key_point_2, dim=-1)

    def manually_set_command(self, cartesian_position_command: torch.Tensor, euler_angle_command: torch.Tensor):
        """Manually set the pose command for the given environment IDs."""
        # check the input shape
        if cartesian_position_command.shape != (self.num_envs, 3):
            raise ValueError(f"cartesian_position_command must have shape ({self.num_envs}, 3), but got {cartesian_position_command.shape}")
        if euler_angle_command.shape != (self.num_envs, 3):
            raise ValueError(f"euler_angle_command must have shape ({self.num_envs}, 3), but got {euler_angle_command.shape}")
        if self.num_envs != cartesian_position_command.shape[0] or self.num_envs != euler_angle_command.shape[0]:
            raise ValueError(f"env_ids must have the same length as cartesian_position_command and euler_angle_command, but got {self.num_envs}, {cartesian_position_command.shape[0]}, {euler_angle_command.shape[0]}")
        # -- position
        self.pose_command_b[:, :3] = cartesian_position_command
        # -- orientation
        quat = quat_from_euler_xyz(*euler_angle_command.unbind(dim=-1))
        # make sure the quaternion has real part as positive
        self.pose_command_b[:, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat

    def _resample_command(self, env_ids: Sequence[int]):
        # sample new pose targets
        # -- position
        r = torch.empty(len(env_ids), device=self.device)
        self.pose_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.pos_x)
        self.pose_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.pos_y)
        self.pose_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.pos_z)
        # -- orientation
        euler_angles = torch.zeros_like(self.pose_command_b[env_ids, :3])
        euler_angles[:, 0].uniform_(*self.cfg.ranges.roll)
        euler_angles[:, 1].uniform_(*self.cfg.ranges.pitch)
        euler_angles[:, 2].uniform_(*self.cfg.ranges.yaw)
        quat = quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])
        # make sure the quaternion has real part as positive
        self.pose_command_b[env_ids, 3:] = quat_unique(quat) if self.cfg.make_quat_unique else quat      

    def _update_command(self):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                # -- goal pose
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.key_point_frame_visualizer_cfg)
                # -- target key points (TODO: change the key point visualizer config for better visualization)
                self.target_key_point_0_visualizer = VisualizationMarkers(self.cfg.key_point_0_visualizer_cfg)
                self.target_key_point_1_visualizer = VisualizationMarkers(self.cfg.key_point_1_visualizer_cfg)
                self.target_key_point_2_visualizer = VisualizationMarkers(self.cfg.key_point_2_visualizer_cfg)
                # -- ee key points (TODO: change the key point visualizer config for better visualization)
                self.ee_key_point_0_visualizer = VisualizationMarkers(self.cfg.key_point_0_visualizer_cfg)
                self.ee_key_point_1_visualizer = VisualizationMarkers(self.cfg.key_point_1_visualizer_cfg)
                self.ee_key_point_2_visualizer = VisualizationMarkers(self.cfg.key_point_2_visualizer_cfg)
                # -- current body pose
                self.current_pose_visualizer = VisualizationMarkers(self.cfg.key_point_frame_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
            self.current_pose_visualizer.set_visibility(True)
            self.target_key_point_0_visualizer.set_visibility(True)
            self.target_key_point_1_visualizer.set_visibility(True)
            self.target_key_point_2_visualizer.set_visibility(True)
            self.ee_key_point_0_visualizer.set_visibility(True)
            self.ee_key_point_1_visualizer.set_visibility(True)
            self.ee_key_point_2_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)
                self.current_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # check if robot is initialized
        # note: this is needed in-case the robot is de-initialized. we can't access the data
        if not self.robot.is_initialized:
            return
        # update the markers
        # -- goal pose
        self.goal_pose_visualizer.visualize(self.pose_command_w[:, :3], self.pose_command_w[:, 3:])
        # -- target key points
        self.target_key_point_0_visualizer.visualize(
            self.pose_command_w_key_point_0[:, :3], self.pose_command_w_key_point_0[:, 3:]
        )
        self.target_key_point_1_visualizer.visualize(
            self.pose_command_w_key_point_1[:, :3], self.pose_command_w_key_point_1[:, 3:]
        )
        self.target_key_point_2_visualizer.visualize(
            self.pose_command_w_key_point_2[:, :3], self.pose_command_w_key_point_2[:, 3:]
        )
        # -- ee key points
        self.ee_key_point_0_visualizer.visualize(
            self.ee_pose_w_key_point_0[:, :3], self.ee_pose_w_key_point_0[:, 3:]
        )
        self.ee_key_point_1_visualizer.visualize(
            self.ee_pose_w_key_point_1[:, :3], self.ee_pose_w_key_point_1[:, 3:]
        )
        self.ee_key_point_2_visualizer.visualize(
            self.ee_pose_w_key_point_2[:, :3], self.ee_pose_w_key_point_2[:, 3:]
        )
        # -- current body pose
        body_link_pose_w = self.robot.data.body_link_pose_w[:, self.body_idx]
        self.current_pose_visualizer.visualize(body_link_pose_w[:, :3], body_link_pose_w[:, 3:7])
