from __future__ import annotations
import torch

from .humanoid_twerk_env_cfg import HumanoidTwerkEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class HumanoidTwerkEnv(LocomotionEnv):
    cfg: HumanoidTwerkEnvCfg

    def __init__(self, cfg: HumanoidTwerkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        device = self.sim.device
        n = self.num_envs

        # twerk variables
        self.pelvis_z0 = torch.zeros(n, device=device)
        self.phase = torch.zeros(n, device=device)

        # indices
        self._pelvis_i = None
        self._rshin = None
        self._lshin = None
        self._rf_body = None
        self._lf_body = None
        self._rfoot_yaw = None
        self._lfoot_yaw = None


    # reset environment index
    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # resolve indices once
        if self._pelvis_i is None:
            body_names = list(self.scene["robot"].data.body_names)
            joint_names = list(self.scene["robot"].data.joint_names)

            self._pelvis_i = body_names.index("pelvis")
            self._rf_body = body_names.index("right_foot")
            self._lf_body = body_names.index("left_foot")
            self._rh = body_names.index("right_hand")
            self._lh = body_names.index("left_hand")
            self._rknee = body_names.index("right_thigh")
            self._lknee = body_names.index("left_thigh")

            self._rshin = joint_names.index("right_shin")
            self._lshin = joint_names.index("left_shin")
            self._rfoot_yaw = joint_names.index("right_foot:1")
            self._lfoot_yaw = joint_names.index("left_foot:1")

        # baseline pelvis height
        pelvis_z = self.robot.data.body_pos_w[env_ids, self._pelvis_i, 2]
        self.pelvis_z0[env_ids] = pelvis_z

        # desync oscillators
        self.phase[env_ids] = 2.0 * torch.pi * torch.rand(len(env_ids), device=self.sim.device)

    # compute values for reward
    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

        # advance twerk oscillator
        self.phase = (self.phase + 2.0 * torch.pi * self.cfg.twerk_freq_hz * self.cfg.sim.dt) % (2.0 * torch.pi)

        # pelvis height
        self.pelvis_z = self.robot.data.body_pos_w[:, self._pelvis_i, 2]

    # reward function
    def _get_rewards(self) -> torch.Tensor:
        actions_cost = torch.sum(self.actions ** 2, dim=-1)

        # check if feet are planted
        th = self.cfg.contact_force_threshold

        fz_l = self.scene["contact_LF"].data.net_forces_w[:, 0, 2].abs()
        fz_r = self.scene["contact_RF"].data.net_forces_w[:, 0, 2].abs()

        c_l = (fz_l > th).float()
        c_r = (fz_r > th).float()
        both_planted = (c_l * c_r) > 0

        # check if the foot slipped (to punish walking)
        lf_v = self.robot.data.body_lin_vel_w[:, self._lf_body, 0:2]
        rf_v = self.robot.data.body_lin_vel_w[:, self._rf_body, 0:2]

        slip = c_l * lf_v.norm(dim=-1) ** 2 + c_r * rf_v.norm(dim=-1) ** 2

        # check angle of knee (encourage bent knees)
        q = self.robot.data.joint_pos
        r = q[:, self._rshin]
        l = q[:, self._lshin]

        knee_err = 0.5 * ((r - self.cfg.knee_target_rad) ** 2 +
                          (l - self.cfg.knee_target_rad) ** 2)
        knee_rew = torch.exp(-self.cfg.knee_k * knee_err)

        # twerking reward
        target_z = self.pelvis_z0 + self.cfg.twerk_amp_m * torch.sin(self.phase)
        twerk_track = torch.exp(-self.cfg.twerk_track_k * (self.pelvis_z - target_z) ** 2)
        twerk_track *= both_planted.float()

        # check stance (encourage wide stance)
        lf_y = self.robot.data.body_pos_w[:, self._lf_body, 1]
        rf_y = self.robot.data.body_pos_w[:, self._rf_body, 1]
        stance_w = torch.abs(lf_y - rf_y)

        width_err = (stance_w - self.cfg.stance_width_target_m) ** 2
        stance_rew = torch.exp(-self.cfg.stance_width_k * width_err)
        stance_rew *= both_planted.float()

        # check contact forces on knees (punish dancing on knees)
        fz_ls = self.scene["contact_LS"].data.net_forces_w[:, 0, 2].abs()
        fz_rs = self.scene["contact_RS"].data.net_forces_w[:, 0, 2].abs()
        shin_contact = ((fz_ls > th) | (fz_rs > th)).float()

        #check foot angle
        q = self.robot.data.joint_pos

        rf = q[:, self._rfoot_yaw]
        lf = q[:, self._lfoot_yaw]

        toe_target = self.cfg.toe_out_rad   # e.g. ±0.35 rad (~20°)

        toe_err = 0.5 * (
            (rf - (-toe_target)) ** 2 +   # right foot outward
            (lf - (+toe_target)) ** 2     # left foot outward
        )

        toe_rew = torch.exp(-self.cfg.toe_k * toe_err)

        #check hands on knees
        hand_r = self.robot.data.body_pos_w[:, self._rh]
        hand_l = self.robot.data.body_pos_w[:, self._lh]
        knee_r = self.robot.data.body_pos_w[:, self._rknee]
        knee_l = self.robot.data.body_pos_w[:, self._lknee]

        # Euclidean distance
        d_r = torch.norm(hand_r - knee_r, dim=-1)
        d_l = torch.norm(hand_l - knee_l, dim=-1)

        # meters, start lenient then tighten later
        tol = self.cfg.hands_tol_m  # start 0.40
        d_avg = 0.5 * (d_r + d_l)

        hands_rew = 1.0 - torch.clamp(d_avg / tol, 0.0, 1.0)
        hands_rew = hands_rew ** 2            # emphasize being really close
        hands_rew *= both_planted.float()

        if not hasattr(self, "_dbg_hands"):
            print("mean dr, dl:", d_r.mean().item(), d_l.mean().item(),
                "min dr, dl:", d_r.min().item(), d_l.min().item())
            self._dbg_hands = True


        # calculate final reward
        reward = (
            self.cfg.alive_reward_scale #reward alive
            + self.cfg.twerk_reward_scale * twerk_track #reward twerking 
            + self.cfg.knee_reward_scale * knee_rew #reward bent knees
            + self.cfg.stance_width_reward_scale * stance_rew #reward wide stance
            + self.cfg.hands_reward_scale * hands_rew #reward hands on knees
            + self.cfg.toe_reward_scale * toe_rew
            - self.cfg.actions_cost_scale * actions_cost #punish action cost 
            - self.cfg.foot_slip_penalty_scale * slip #punish foot slip
            - self.cfg.shin_contact_penalty * shin_contact #punish on knees
        )

        return torch.where(
            self.reset_terminated,
            torch.full_like(reward, self.cfg.death_cost),
            reward,
        )
