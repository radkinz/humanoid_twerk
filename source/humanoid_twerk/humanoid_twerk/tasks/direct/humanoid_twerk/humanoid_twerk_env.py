from __future__ import annotations
import torch

from .humanoid_twerk_env_cfg import HumanoidTwerkEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class HumanoidTwerkEnv(LocomotionEnv):
    cfg: HumanoidTwerkEnvCfg

    def __init__(self, cfg: HumanoidTwerkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.air_time = torch.zeros(self.num_envs, device=self.sim.device)
        #twerk logic
        self.pelvis_z0 = torch.zeros(self.num_envs, device=self.sim.device)
        self.phase = torch.zeros(self.num_envs, device=self.sim.device)
        self._pelvis_i = None
        print(self.robot.data.joint_names)
        #knee joints to make sure they are bent
        self._rshin = None
        self._lshin = None


    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # pelvis, right & left knee indexing
        if self._pelvis_i is None:
            names = list(self.scene["robot"].data.body_names)
            self._pelvis_i = names.index("pelvis")
            self._rshin = names.index("right_shin")
            self._lshin = names.index("left_shin")

        # current pelvis height
        pelvis_z = self.robot.data.body_pos_w[env_ids, self._pelvis_i, 2]
        self.pelvis_z0[env_ids] = pelvis_z

        # randomize phase so all envs aren't synchronized
        self.phase[env_ids] = 2.0 * torch.pi * torch.rand(len(env_ids), device=self.sim.device)

        self.air_time[env_ids] = 0.0

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

        #twerk checker
        # advance phase (same for all envs; you can randomize at reset later)
        freq_hz = getattr(self.cfg, "twerk_freq_hz", 2.5)
        self.phase = (self.phase + 2.0 * torch.pi * freq_hz * self.cfg.sim.dt) % (2.0 * torch.pi)

        pelvis_z = self.robot.data.body_pos_w[:, self._pelvis_i, 2]
        self.pelvis_z = pelvis_z

    
    def _get_rewards(self) -> torch.Tensor:
        """Twerk reward: track a sinusoidal pelvis-height motion while keeping both feet planted and knees bent."""
        actions_cost = torch.sum(self.actions ** 2, dim=-1)

        # --- contact forces (feet planted gate) ---
        fz_l = self.scene["contact_LF"].data.net_forces_w[:, 0, 2].abs()
        fz_r = self.scene["contact_RF"].data.net_forces_w[:, 0, 2].abs()
        th = self.cfg.contact_force_threshold
        both_planted = (fz_l > th) & (fz_r > th)  # (num_envs,)

        # --- knees planted
        q = self.robot.data.joint_pos  # (num_envs, num_joints)

        r = q[:, self._rshin]
        l = q[:, self._lshin]

        # pick a target bend; SIGN may be + or - depending on the model
        target = self.cfg.knee_target_rad

        knee_err = 0.5 * ((r - target) ** 2 + (l - target) ** 2)
        knee_rew = torch.exp(-self.cfg.knee_k * knee_err)   # in (0, 1]

        # --- pelvis height signal ---
        # pelvis index (once)
        if not hasattr(self, "_pelvis_i"):
            names = list(self.scene["robot"].data.body_names)
            self._pelvis_i = names.index("pelvis")

        # pelvis z (prefer body_pos_w)
        if hasattr(self.robot.data, "body_pos_w"):
            pelvis_z = self.robot.data.body_pos_w[:, self._pelvis_i, 2]
        elif hasattr(self.robot.data, "body_state_w"):
            # common layout: (..., 13) with pos in [:, :, 0:3]
            pelvis_z = self.robot.data.body_state_w[:, self._pelvis_i, 2]
        else:
            raise RuntimeError(
                "Couldn't find body positions on robot.data. "
                "Expected 'body_pos_w' or 'body_state_w'. "
                "Print(dir(self.robot.data)) to locate body pose fields."
            )
        self.pelvis_z = pelvis_z

        # baseline pelvis height (once; should also be reset in _reset_idx for best results)
        if not hasattr(self, "pelvis_z0"):
            self.pelvis_z0 = pelvis_z.clone()

        # --- phase oscillator (can live here for simplicity) ---
        if not hasattr(self, "phase"):
            self.phase = torch.zeros(self.num_envs, device=self.sim.device)
        self.phase = (self.phase + 2.0 * torch.pi * self.cfg.twerk_freq_hz * self.cfg.sim.dt) % (2.0 * torch.pi)

        # --- twerk tracking reward ---
        target = self.pelvis_z0 + self.cfg.twerk_amp_m * torch.sin(self.phase)
        # exp(-k * error^2) gives smooth reward in [0, 1]
        twerk_track = torch.exp(-self.cfg.twerk_track_k * (pelvis_z - target) ** 2)
        # only count it when both feet are planted
        twerk_track = twerk_track * both_planted.float()

        # --- final reward ---
        reward = (
            self.cfg.alive_reward_scale
            + self.cfg.twerk_reward_scale * twerk_track #reward moving pelvis
            - self.cfg.jump_actions_cost_scale * actions_cost #punish jumping aka reward feet planted
            + self.cfg.knee_reward_scale * knee_rew #reward knees planted
        )

        # death / terminated handling
        reward = torch.where(
            self.reset_terminated,
            torch.full_like(reward, self.cfg.death_cost),
            reward,
        )

        return reward

