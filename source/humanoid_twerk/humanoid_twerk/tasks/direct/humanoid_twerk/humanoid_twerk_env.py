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

    def _height(self, env_ids=None):
        # Prefer torso position if the base env exposes it
        if hasattr(self, "torso_position"):
            return self.torso_position[:, 2] if env_ids is None else self.torso_position[env_ids, 2]
        # Fallback: robot root height
        return self.robot.data.root_pos_w[:, 2] if env_ids is None else self.robot.data.root_pos_w[env_ids, 2]

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)

        self.air_time[env_ids] = 0.0

        h = self._height(env_ids)
        self.stand_height[env_ids] = h
        self.peak_height[env_ids] = h
        self.prev_height[env_ids] = h  # avoids a big first-step spike

    def _compute_intermediate_values(self):
        super()._compute_intermediate_values()

        self.curr_height = self._height()
        self.peak_height = torch.maximum(self.peak_height, self.curr_height)

  
        # ContactSensor forces: (num_envs, 1, 3)
        fz_l = self.scene["contact_LF"].data.net_forces_w[:, 0, 2].abs()
        fz_r = self.scene["contact_RF"].data.net_forces_w[:, 0, 2].abs()

        thresh = self.cfg.contact_force_threshold
        both_off = (fz_l < thresh) & (fz_r < thresh)

        self.air_time = torch.where(
            both_off,
            self.air_time + self.cfg.sim.dt,
            torch.zeros_like(self.air_time),
        )

    def _get_rewards(self) -> torch.Tensor:
        jump_progress = (self.curr_height - self.prev_height) / self.cfg.sim.dt
        actions_cost = torch.sum(self.actions ** 2, dim=-1)

        jump_height = torch.clamp(self.peak_height - self.stand_height - 0.10, 0.0, 0.6) / 0.6
        air_rew = torch.clamp(self.air_time, 0.0, 0.5) / 0.5

        reward = (
            self.cfg.height_reward_scale * self.curr_height
            + self.cfg.jump_progress_scale * jump_progress
            + self.cfg.jump_height_reward_scale * jump_height
            - self.cfg.jump_actions_cost_scale * actions_cost
            + self.cfg.alive_reward_scale
            + self.cfg.air_time_reward_scale * air_rew
        )

        reward = torch.where(
            self.reset_terminated,
            torch.full_like(reward, self.cfg.death_cost),
            reward,
        )

        self.prev_height[:] = self.curr_height
        return reward
