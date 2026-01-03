from __future__ import annotations
import torch

from .humanoid_twerk_env_cfg import HumanoidTwerkEnvCfg
from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


class HumanoidTwerkEnv(LocomotionEnv):
    cfg: HumanoidTwerkEnvCfg

    def __init__(self, cfg: HumanoidTwerkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.prev_height = torch.zeros(self.num_envs, device=self.sim.device)
        self.stand_height = torch.zeros(self.num_envs, device=self.sim.device)
        self.peak_height = torch.zeros(self.num_envs, device=self.sim.device)
        self.air_time = torch.zeros(self.num_envs, device=self.sim.device)
        #twerk logic
        self.pelvis_z0 = torch.zeros(self.num_envs, device=self.sim.device)
        self.phase = torch.zeros(self.num_envs, device=self.sim.device)
        self._pelvis_i = None



    def _height(self, env_ids=None):
        # Prefer torso position if the base env exposes it
        if hasattr(self, "torso_position"):
            return self.torso_position[:, 2] if env_ids is None else self.torso_position[env_ids, 2]
        # Fallback: robot root height
        return self.robot.data.root_pos_w[:, 2] if env_ids is None else self.robot.data.root_pos_w[env_ids, 2]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        # pelvis index once
        if self._pelvis_i is None:
            names = list(self.scene["robot"].data.body_names)
            self._pelvis_i = names.index("pelvis")

        # current pelvis height
        pelvis_z = self.robot.data.body_pos_w[env_ids, self._pelvis_i, 2]
        self.pelvis_z0[env_ids] = pelvis_z

        # randomize phase so all envs aren't synchronized
        self.phase[env_ids] = 2.0 * torch.pi * torch.rand(len(env_ids), device=self.sim.device)


        self.air_time[env_ids] = 0.0

        h = self._height(env_ids)
        self.stand_height[env_ids] = h
        self.peak_height[env_ids] = h
        self.prev_height[env_ids] = h  # avoids a big first-step spike

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

        self.curr_height = self._height()
        self.peak_height = torch.maximum(self.peak_height, self.curr_height)

        #twerk checker
        # advance phase (same for all envs; you can randomize at reset later)
        freq_hz = getattr(self.cfg, "twerk_freq_hz", 2.5)
        self.phase = (self.phase + 2.0 * torch.pi * freq_hz * self.cfg.sim.dt) % (2.0 * torch.pi)

        pelvis_z = self.robot.data.body_pos_w[:, self._pelvis_i, 2]
        self.pelvis_z = pelvis_z

    
        # ContactSensor forces: (num_envs, 1, 3)
        fz_l = self.scene["contact_LF"].data.net_forces_w[:, 0, 2].abs()
        fz_r = self.scene["contact_RF"].data.net_forces_w[:, 0, 2].abs()
        th = self.cfg.contact_force_threshold
        both_planted = (fz_l > th) & (fz_r > th) #higher the force then the feet are on the ground triggering sensor

        self.air_time = torch.where(
            both_planted,
            self.air_time + self.cfg.sim.dt,
            torch.zeros_like(self.air_time),
        )
    
    def _get_rewards(self) -> torch.Tensor:
        """Twerk reward: track a sinusoidal pelvis-height motion while keeping BOTH feet planted."""
        # --- basic terms you already had ---
        actions_cost = torch.sum(self.actions ** 2, dim=-1)

        # keep these for stability / debugging
        if not hasattr(self, "curr_height"):
            # fallback if you didn't set curr_height elsewhere
            if hasattr(self, "torso_position"):
                self.curr_height = self.torso_position[:, 2]
            else:
                self.curr_height = self.robot.data.root_pos_w[:, 2]

        # --- contact forces (feet planted gate) ---
        fz_l = self.scene["contact_LF"].data.net_forces_w[:, 0, 2].abs()
        fz_r = self.scene["contact_RF"].data.net_forces_w[:, 0, 2].abs()
        th = self.cfg.contact_force_threshold
        both_planted = (fz_l > th) & (fz_r > th)  # (num_envs,)

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
            + self.cfg.height_reward_scale * self.curr_height
            + self.cfg.twerk_reward_scale * twerk_track
            - self.cfg.jump_actions_cost_scale * actions_cost
        )

        # death / terminated handling
        reward = torch.where(
            self.reset_terminated,
            torch.full_like(reward, self.cfg.death_cost),
            reward,
        )

        # keep prev_height updated if you still use it elsewhere
        if hasattr(self, "prev_height"):
            self.prev_height[:] = self.curr_height

        return reward

