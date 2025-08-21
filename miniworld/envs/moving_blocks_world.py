from gymnasium import utils
import math
import numpy as np

from miniworld.entity import Box, COLOR_NAMES
from miniworld.envs.putnext import PutNext
from miniworld.math import intersect_circle_segs
from miniworld.miniworld import MiniWorldEnv


class MovingBlocksWorld(PutNext, utils.EzPickle):
    """
    Single-room environment derived from PutNext where colored boxes move
    autonomously each step. Reward-based termination is disabled: episodes only
    end due to time limits.

    Options:
    - box_speed_scale: scales continuous box speed when not in grid mode
    - box_allow_overlap: if True, boxes ignore box-box collisions
    - agent_box_allow_overlap: if True, agent and boxes ignore collisions
    - box_random_orientation: if True, keep randomized box orientations on reset;
      otherwise orientations are aligned to 0
    - grid_mode: if True, snap agent/boxes to integer grid and use integer
      per-axis velocities for boxes
    - grid_vel_min / grid_vel_max: integer velocity component range for boxes in
      grid mode (inclusive). (0,0) is avoided
    """

    def __init__(
        self,
        size=12,
        box_speed_scale=1.0,
        box_allow_overlap=False,
        agent_box_allow_overlap=False,
        box_random_orientation=False,
        grid_mode=False,
        grid_vel_min=-1,
        grid_vel_max=1,
        num_blocks=6,
        allow_color_repeat=False,
        color_pool=None,
        near_margin=None,
        **kwargs,
    ):
        self.box_speed_scale = float(box_speed_scale)
        self.box_allow_overlap = bool(box_allow_overlap)
        self.agent_box_allow_overlap = bool(agent_box_allow_overlap)
        self.box_random_orientation = bool(box_random_orientation)
        self.grid_mode = bool(grid_mode)
        self.grid_vel_min = int(grid_vel_min)
        self.grid_vel_max = int(grid_vel_max)
        self.num_blocks = int(num_blocks)
        self.allow_color_repeat = bool(allow_color_repeat)
        self.color_pool = list(color_pool) if color_pool is not None else list(COLOR_NAMES)
        super().__init__(size=size, **kwargs)
        # Keep for compatibility; not used for reward anymore
        self.near_margin = (
            float(near_margin)
            if near_margin is not None
            else 1.1 * float(self.max_forward_step)
        )
        utils.EzPickle.__init__(
            self,
            size,
            box_speed_scale,
            box_allow_overlap,
            agent_box_allow_overlap,
            box_random_orientation,
            self.grid_mode,
            self.grid_vel_min,
            self.grid_vel_max,
            self.num_blocks,
            self.allow_color_repeat,
            self.color_pool,
            self.near_margin,
            **kwargs,
        )

    def _gen_world(self):
        # Create a rectangular room
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size)

        # Choose colors for the blocks
        if not self.allow_color_repeat and self.num_blocks <= len(self.color_pool):
            chosen_colors = list(self.np_random.choice(self.color_pool, size=self.num_blocks, replace=False))
        else:
            chosen_colors = list(self.np_random.choice(self.color_pool, size=self.num_blocks, replace=True))

        # Place blocks with random sizes
        for color in chosen_colors:
            box = Box(color=color, size=self.np_random.uniform(0.6, 0.85))
            self.place_entity(box)

        # Place the agent at a random position
        self.place_agent()

    def _quantize_heading(self):
        q = (math.pi / 2)
        self.agent.dir = round(self.agent.dir / q) * q
        if self.agent.carrying is not None:
            self.agent.carrying.dir = self.agent.dir

    def _snap_entity_to_grid(self, ent):
        snapped = ent.pos.copy()
        snapped[0] = round(snapped[0])
        snapped[2] = round(snapped[2])
        if not self.intersect(ent, snapped, ent.radius):
            ent.pos = snapped
            return
        for dx in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                candidate = snapped.copy()
                candidate[0] += dx
                candidate[2] += dz
                if not self.intersect(ent, candidate, ent.radius):
                    ent.pos = candidate
                    return

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        if not self.box_random_orientation:
            for ent in self.entities:
                if isinstance(ent, Box):
                    ent.dir = 0.0

        if self.grid_mode:
            self._snap_entity_to_grid(self.agent)
            for ent in self.entities:
                if isinstance(ent, Box):
                    self._snap_entity_to_grid(ent)

        # Assign velocities
        rand = self.np_random if self.domain_rand else None
        move_step = self.params.sample(rand, "forward_step") * self.box_speed_scale

        for ent in self.entities:
            if not isinstance(ent, Box):
                continue
            if self.grid_mode:
                while True:
                    vx = self.np_random.integers(self.grid_vel_min, self.grid_vel_max + 1)
                    vz = self.np_random.integers(self.grid_vel_min, self.grid_vel_max + 1)
                    if vx != 0 or vz != 0:
                        break
                ent.velocity = np.array([int(vx), 0.0, int(vz)], dtype=float)
            else:
                theta = self.np_random.uniform(-math.pi, math.pi)
                vx = move_step * math.cos(theta)
                vz = move_step * math.sin(theta)
                ent.velocity = np.array([vx, 0.0, vz], dtype=float)

        return obs, info

    def intersect(self, ent, pos, radius):
        # Ignore Y
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Entities
        for ent2 in self.entities:
            if ent2 is ent:
                continue
            if self.box_allow_overlap and isinstance(ent, Box) and isinstance(ent2, Box):
                continue
            if self.agent_box_allow_overlap and (
                (ent is self.agent and isinstance(ent2, Box))
                or (ent2 is self.agent and isinstance(ent, Box))
            ):
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])
            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def step(self, action):
        # Avoid PutNext's reward-based termination by calling MiniWorldEnv.step
        if self.grid_mode:
            original_params = self.params
            params_copy = self.params.copy()
            params_copy.set("forward_drift", 0.0, 0.0, 0.0)
            self.params = params_copy
            obs, reward, termination, truncation, info = MiniWorldEnv.step(self, action)
            self.params = original_params
        else:
            obs, reward, termination, truncation, info = MiniWorldEnv.step(self, action)

        if termination or truncation:
            return obs, reward, termination, truncation, info

        if self.grid_mode:
            self._snap_entity_to_grid(self.agent)
            self._quantize_heading()
            if self.agent.carrying is not None:
                self._snap_entity_to_grid(self.agent.carrying)

        carrying = self.agent.carrying

        for ent in list(self.entities):
            if not isinstance(ent, Box):
                continue
            if carrying is not None and ent is carrying:
                continue

            if not hasattr(ent, "velocity"):
                theta = self.np_random.uniform(-math.pi, math.pi)
                rand = self.np_random if self.domain_rand else None
                move_step = self.params.sample(rand, "forward_step") * self.box_speed_scale
                ent.velocity = np.array(
                    [move_step * math.cos(theta), 0.0, move_step * math.sin(theta)],
                    dtype=float,
                )

            current_pos = ent.pos

            # X axis
            if ent.velocity[0] != 0.0:
                candidate = current_pos.copy()
                candidate[0] += ent.velocity[0]
                hit = self.intersect(ent, candidate, ent.radius)
                if (not hit) or (self.box_allow_overlap and isinstance(hit, Box)):
                    current_pos = candidate
                else:
                    ent.velocity[0] = -ent.velocity[0]
                    candidate_bounce = current_pos.copy()
                    candidate_bounce[0] += ent.velocity[0]
                    hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                    if (not hit_bounce) or (
                        self.box_allow_overlap and isinstance(hit_bounce, Box)
                    ):
                        current_pos = candidate_bounce
            if self.grid_mode:
                current_pos[0] = round(current_pos[0])

            # Z axis
            if ent.velocity[2] != 0.0:
                candidate = current_pos.copy()
                candidate[2] += ent.velocity[2]
                hit = self.intersect(ent, candidate, ent.radius)
                if (not hit) or (self.box_allow_overlap and isinstance(hit, Box)):
                    current_pos = candidate
                else:
                    ent.velocity[2] = -ent.velocity[2]
                    candidate_bounce = current_pos.copy()
                    candidate_bounce[2] += ent.velocity[2]
                    hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                    if (not hit_bounce) or (
                        self.box_allow_overlap and isinstance(hit_bounce, Box)
                    ):
                        current_pos = candidate_bounce
            if self.grid_mode:
                current_pos[2] = round(current_pos[2])

            ent.pos = current_pos

        # No reward-based termination condition here
        return obs, reward, termination, truncation, info


