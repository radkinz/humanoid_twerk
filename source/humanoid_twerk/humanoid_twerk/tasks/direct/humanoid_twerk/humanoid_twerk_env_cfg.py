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


# ----------------------------
# Shared helpers / constants
# ----------------------------

ENV_ROBOT = "/World/envs/env_.*/Robot"


WIDE_SQUAT_INIT = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 1.25),
    joint_pos={
        # wide stance
        "right_thigh:0": -0.25,
        "left_thigh:0":  -0.25,
        # bend knees slightly
        "right_shin": -0.5,
        "left_shin":  -0.5,

        # shoulders forward
        "right_upper_arm:0": 0.9,
        "left_upper_arm:0":  0.9,
        # bend elbows
        "right_lower_arm": -1.2,
        "left_lower_arm":  -1.2,
    },
)


def make_robot_cfg() -> ArticulationCfg:
    """One canonical robot cfg so scene.robot and cfg.robot can't drift."""
    return HUMANOID_CFG.replace(prim_path=ENV_ROBOT, init_state=WIDE_SQUAT_INIT)


def make_contact_sensor(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{ENV_ROBOT}/{body_name}",
        update_period=0.0,
    )


# ----------------------------
# Scene
# ----------------------------

@configclass
class HumanoidTwerkSceneCfg(InteractiveSceneCfg):
    """Robot + contact sensors only (LocomotionEnv spawns terrain from cfg.terrain)."""

    robot: ArticulationCfg = make_robot_cfg()

    # Declare sensors via a small mapping
    contact_LF: ContactSensorCfg = make_contact_sensor("left_foot")
    contact_RF: ContactSensorCfg = make_contact_sensor("right_foot")
    contact_LS: ContactSensorCfg = make_contact_sensor("left_shin")
    contact_RS: ContactSensorCfg = make_contact_sensor("right_shin")


# ----------------------------
# Base env cfg
# ----------------------------
@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 2
    action_scale = 1.0
    action_space = 21
    observation_space = 75
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

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

    scene: HumanoidTwerkSceneCfg = HumanoidTwerkSceneCfg(
        num_envs=1024,
        env_spacing=4.0,
        replicate_physics=False,
        clone_in_fabric=False,
    )

    robot: ArticulationCfg = make_robot_cfg()

    joint_gears: list = [
        67.5000, 67.5000,
        67.5000, 67.5000,
        67.5000, 67.5000,
        67.5000,
        45.0000, 45.0000,
        45.0000, 135.0000, 45.0000,
        45.0000, 135.0000, 45.0000,
        90.0000, 90.0000,
        22.5, 22.5, 22.5, 22.5,
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


# ----------------------------
# Twerk task cfg
# ----------------------------

@configclass
class HumanoidTwerkEnvCfg(HumanoidEnvCfg):
    # twerk objective
    twerk_reward_scale: float = 2.0
    twerk_freq_hz: float = 2.5
    twerk_amp_m: float = 0.08
    twerk_track_k: float = 10.0

    # contact / penalties
    contact_force_threshold: float = 5.0
    actions_cost_scale: float = 0.005
    foot_slip_penalty_scale: float = 2.0
    shin_contact_penalty: float = 2.0

    # knees + stance shaping
    knee_reward_scale: float = 1.0
    knee_target_rad: float = -0.5
    knee_k: float = 8.0

    #stance
    stance_width_reward_scale: float = 1.0
    stance_width_target_m: float = 0.35
    stance_width_k: float = 30.0
    stance_width_min_m: float = 0.25
    stance_min_reward_scale: float = 0.5

    #toes
    toe_reward_scale: float = 0.5
    toe_out_rad: float = 0.35
    toe_k: float = 10.0

    #hands
    hands_reward_scale: float = 4.0
    hands_k: float = 20.0   # higher = tighter hand placement
    hands_tol_m: float = 0.40   
    pelvis_min_height: float = 0.95  # tune; squat should still be above this

    def __post_init__(self):
        super().__post_init__()
        # Enable PhysX contact reporting
        self.robot.spawn.activate_contact_sensors = True
        self.scene.robot.spawn.activate_contact_sensors = True
