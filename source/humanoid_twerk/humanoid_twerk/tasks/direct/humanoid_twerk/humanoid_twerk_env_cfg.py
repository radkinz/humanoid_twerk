# Copyright (c) 2022-2026, The Isaac Lab Project Developers
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab_assets import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg


@configclass
class HumanoidTwerkSceneCfg(InteractiveSceneCfg):
    """Scene containing robot + contact sensors.

    NOTE: Do NOT put terrain here for direct LocomotionEnv, because LocomotionEnv
    spawns terrain from cfg.terrain and you'll get duplicate prim errors.
    """

    # robot (also kept at top-level cfg.robot for compatibility with base env)
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    contact_LF: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/left_foot",
        update_period=0.0,
    )
    contact_RF: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/right_foot",
        update_period=0.0,
    )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Base humanoid direct RL env config (keeps the fields LocomotionEnv expects)."""

    # env
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # terrain (LocomotionEnv expects cfg.terrain and will spawn it)
    terrain: TerrainImporterCfg = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene (robot + sensors)
    scene: HumanoidTwerkSceneCfg = HumanoidTwerkSceneCfg(
        num_envs=1024,          # start lower while debugging
        env_spacing=4.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # robot (some direct env code references cfg.robot, so keep it here too)
    robot: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # misc humanoid params (kept from your original cfg)
    joint_gears: list = [
        67.5000,  # lower_waist
        67.5000,  # lower_waist
        67.5000,  # right_upper_arm
        67.5000,  # right_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # left_upper_arm
        67.5000,  # pelvis
        45.0000,  # right_lower_arm
        45.0000,  # left_lower_arm
        45.0000,  # right_thigh: x
        135.0000,  # right_thigh: y
        45.0000,  # right_thigh: z
        45.0000,  # left_thigh: x
        135.0000,  # left_thigh: y
        45.0000,  # left_thigh: z
        90.0000,  # right_knee
        90.0000,  # left_knee
        22.5,  # right_foot
        22.5,  # right_foot
        22.5,  # left_foot
        22.5,  # left_foot
    ]

    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = 0.8

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.01


@configclass
class HumanoidTwerkEnvCfg(HumanoidEnvCfg):
    """Custom humanoid env cfg for jump/airtime shaping (pre-twerk)."""
    twerk_reward_scale: float = 2.0
    twerk_freq_hz: float = 2.5
    twerk_amp_m: float = 0.08
    twerk_track_k: float = 10.0
    contact_force_threshold: float = 5.0
    jump_actions_cost_scale: float = 0.005
    alive_reward_scale: float = 2.0
    death_cost: float = -1.0
    # knee-bend (shin joints) shaping
    knee_reward_scale: float = 1.0
    knee_target_rad: float = 0.8   # if bends wrong way, flip to -0.8
    knee_k: float = 8.0

    def __post_init__(self):
        super().__post_init__()

        # Enable PhysX contact reporting so ContactSensor outputs forces.
        # Set on BOTH cfg.robot and cfg.scene.robot for compatibility.
        self.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.activate_contact_sensors = True
