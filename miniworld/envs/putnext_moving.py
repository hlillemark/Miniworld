from gymnasium import utils
import math
import numpy as np
from miniworld.math import intersect_circle_segs

from miniworld.entity import Box
from miniworld.envs.putnext import PutNext


class PutNextMoving(PutNext, utils.EzPickle):
    """
    ## Description

    Single-room environment like PutNext where a red box must be placed next
    to a yellow box. Additionally, each box is assigned a random 2D velocity at
    reset time and will move by that velocity at every step (unless being
    carried). If a move would collide with a wall or another entity, the box's
    velocity is reversed (bounces), and the box attempts the reversed move once.

    Optional behavior:
    - If `box_allow_overlap=True`, boxes ignore collisions with other boxes
      (they only collide with walls and the agent).
    - If `agent_box_allow_overlap=True`, the agent and boxes ignore collisions
      with each other (both ways). Walls still collide.

    ## Action Space

    Same as `PutNext`.

    ## Observation Space

    Same as `PutNext`.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box is next to yellow box

    ## Arguments

    * `size`: size of world

    ```python
    env = gymnasium.make("MiniWorld-PutNextMoving-v0", size=12)
    ```
    """

    def __init__(self, size=12, box_speed_scale=1.0, box_allow_overlap=False, agent_box_allow_overlap=False, box_random_orientation=False, near_margin=None, **kwargs):
        self.box_speed_scale = float(box_speed_scale)
        self.box_allow_overlap = bool(box_allow_overlap)
        self.agent_box_allow_overlap = bool(agent_box_allow_overlap)
        self.box_random_orientation = bool(box_random_orientation)
        super().__init__(size=size, **kwargs)
        # Decouple success tolerance from forward_step; default to the base max_forward_step at init
        self.near_margin = (
            float(near_margin)
            if near_margin is not None
            else 1.1 * float(self.max_forward_step)
        )
        utils.EzPickle.__init__(self, size, box_speed_scale, box_allow_overlap, agent_box_allow_overlap, box_random_orientation, self.near_margin, **kwargs)

    def intersect(self, ent, pos, radius):
        """
        Intersect that optionally ignores box-box or agent-box collisions based on flags.
        Returns True for wall collisions, the colliding entity for entity collisions, or None.
        """

        # Ignore the Y position
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        # Check for intersection with walls
        if intersect_circle_segs(pos, radius, self.wall_segs):
            return True

        # Check for entity intersection
        for ent2 in self.entities:
            if ent2 is ent:
                continue

            # Apply overlap options
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

    def _boxes_next_to_each_other(self):
        d = np.linalg.norm(self.red_box.pos - self.yellow_box.pos)
        return d < (self.red_box.radius + self.yellow_box.radius + self.near_margin)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)

        # Apply box orientation policy (default aligned, otherwise randomized already by spawning)
        if not self.box_random_orientation:
            for ent in self.entities:
                if isinstance(ent, Box):
                    ent.dir = 0.0

        # Assign a random velocity to each box at episode start
        rand = self.np_random if self.domain_rand else None
        move_step = self.params.sample(rand, "forward_step") * self.box_speed_scale

        for ent in self.entities:
            if not isinstance(ent, Box):
                continue
            theta = self.np_random.uniform(-math.pi, math.pi)
            vx = move_step * math.cos(theta)
            vz = move_step * math.sin(theta)
            ent.velocity = np.array([vx, 0.0, vz], dtype=float)

        return obs, info

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # If episode already ended in the base step, don't move boxes
        if termination or truncation:
            return obs, reward, termination, truncation, info

        # After applying the agent action, move each box along its own velocity
        carrying = self.agent.carrying

        for ent in list(self.entities):
            if not isinstance(ent, Box):
                continue
            if carrying is not None and ent is carrying:
                continue

            # Ensure velocity exists (in case of custom resets)
            if not hasattr(ent, "velocity"):
                theta = self.np_random.uniform(-math.pi, math.pi)
                rand = self.np_random if self.domain_rand else None
                move_step = self.params.sample(rand, "forward_step") * self.box_speed_scale
                ent.velocity = np.array(
                    [move_step * math.cos(theta), 0.0, move_step * math.sin(theta)],
                    dtype=float,
                )

            # Axis-separated movement with per-axis bounce
            current_pos = ent.pos

            # Move along X axis first
            if ent.velocity[0] != 0.0:
                candidate = current_pos.copy()
                candidate[0] += ent.velocity[0]
                hit = self.intersect(ent, candidate, ent.radius)
                if (not hit) or (self.box_allow_overlap and isinstance(hit, Box)):
                    current_pos = candidate
                else:
                    # Reverse X velocity and attempt bounce move along X
                    ent.velocity[0] = -ent.velocity[0]
                    candidate_bounce = current_pos.copy()
                    candidate_bounce[0] += ent.velocity[0]
                    hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                    if (not hit_bounce) or (
                        self.box_allow_overlap and isinstance(hit_bounce, Box)
                    ):
                        current_pos = candidate_bounce

            # Then move along Z axis
            if ent.velocity[2] != 0.0:
                candidate = current_pos.copy()
                candidate[2] += ent.velocity[2]
                hit = self.intersect(ent, candidate, ent.radius)
                if (not hit) or (self.box_allow_overlap and isinstance(hit, Box)):
                    current_pos = candidate
                else:
                    # Reverse Z velocity and attempt bounce move along Z
                    ent.velocity[2] = -ent.velocity[2]
                    candidate_bounce = current_pos.copy()
                    candidate_bounce[2] += ent.velocity[2]
                    hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                    if (not hit_bounce) or (
                        self.box_allow_overlap and isinstance(hit_bounce, Box)
                    ):
                        current_pos = candidate_bounce

            ent.pos = current_pos

        # Re-evaluate success condition after boxes moved (using fixed near margin)
        if not self.agent.carrying and self._boxes_next_to_each_other():
            reward += self._reward()
            termination = True

        return obs, reward, termination, truncation, info


