from gymnasium import utils
import math
import numpy as np

from miniworld.entity import Box, Ball, COLOR_NAMES
from miniworld.envs.putnext import PutNext
from miniworld.math import intersect_circle_segs
from miniworld.miniworld import MiniWorldEnv


class MovingBlockWorld(PutNext, utils.EzPickle):
    """
    Single-room environment derived from PutNext where colored boxes move
    autonomously each step. Reward-based termination is disabled: episodes only
    end due to time limits.

    Options:
    - floor_tex: texture name for the floor (default "white")
    - wall_tex: texture name for the walls (default "white")
    - ceil_tex: texture name for the ceiling (default "white")
    - box_speed_scale: scales continuous box speed when not in grid mode
    - box_allow_overlap: if True, boxes ignore box-box collisions
    - agent_box_allow_overlap: if True, agent and boxes ignore collisions
    - box_random_orientation: if True, keep randomized box orientations on reset;
      otherwise orientations are aligned to 0
    - grid_mode: if True, snap agent/boxes to integer grid and use integer
      per-axis velocities for boxes
    - grid_vel_min / grid_vel_max: integer velocity component range for boxes in
      grid mode (inclusive). (0,0) is avoided
    - block_torus_wrap: if True, boxes wrap across walls (torus); agent unchanged
    """

    def __init__(
        self,
        size=12,
        floor_tex="concrete",
        wall_tex="white",
        ceil_tex="ceiling_tile_noborder",
        box_tex=None,
        box_tex_randomize=False,
        wall_tex_randomize=False,
        floor_tex_randomize=False,
        box_and_ball=False,
        box_speed_scale=1.0,
        box_allow_overlap=False,
        agent_box_allow_overlap=False,
        box_random_orientation=False,
        blocks_static=False,
        block_torus_wrap=False,
        spawn_wall_buffer=None,
        grid_mode=False,
        grid_vel_min=-1,
        grid_vel_max=1,
        grid_cardinal_only=False,
        num_blocks=6,
        allow_color_repeat=False,
        color_pool=None,
        ensure_base_palette=False,
        near_margin=None,
        block_size_xy=None,
        block_height=None,
        agent_center_start=False,
        **kwargs,
    ):
        self.box_speed_scale = float(box_speed_scale)
        self.box_allow_overlap = bool(box_allow_overlap)
        self.agent_box_allow_overlap = bool(agent_box_allow_overlap)
        self.box_random_orientation = bool(box_random_orientation)
        self.blocks_static = bool(blocks_static)
        self.block_torus_wrap = bool(block_torus_wrap)
        self.grid_mode = bool(grid_mode)
        self.spawn_wall_buffer = float(spawn_wall_buffer) if spawn_wall_buffer is not None else None
        self.grid_vel_min = int(grid_vel_min)
        self.grid_vel_max = int(grid_vel_max)
        self.grid_cardinal_only = bool(grid_cardinal_only)
        self.num_blocks = int(num_blocks)
        self.allow_color_repeat = bool(allow_color_repeat)
        self.color_pool = list(color_pool) if color_pool is not None else list(COLOR_NAMES)
        self.ensure_base_palette = bool(ensure_base_palette)
        # Optional uniform block sizing controls
        # If provided, all blocks will use (block_size_xy, block_height, block_size_xy)
        self.block_size_xy = None if block_size_xy is None else float(block_size_xy)
        self.block_height = None if block_height is None else float(block_height)
        self.agent_center_start = bool(agent_center_start)
        # Store texture overrides
        self._floor_tex_override = str(floor_tex) if floor_tex is not None else None
        self._wall_tex_override = str(wall_tex) if wall_tex is not None else None
        self._ceil_tex_override = str(ceil_tex) if ceil_tex is not None else None
        self._box_tex_override = str(box_tex) if box_tex is not None else None
        self.box_tex_randomize = bool(box_tex_randomize)
        # Default pool for randomized box textures
        self._box_tex_pool = ["ceiling_tiles", "airduct_grate", "checkerboard"]
        # Randomize room textures
        self.wall_tex_randomize = bool(wall_tex_randomize)
        self.floor_tex_randomize = bool(floor_tex_randomize)
        # Default pools for walls/floors
        self._wall_tex_pool = ["brick_wall", "wood_planks", "wood"]
        self._floor_tex_pool = ["cardboard", "grass", "concrete"]
        # Randomize between box and ball per entity
        self.box_and_ball = bool(box_and_ball)
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
            floor_tex,
            wall_tex,
            ceil_tex,
            blocks_static,
            spawn_wall_buffer,
            box_speed_scale,
            box_allow_overlap,
            agent_box_allow_overlap,
            box_random_orientation,
            self.grid_mode,
            self.grid_vel_min,
            self.grid_vel_max,
            self.grid_cardinal_only,
            self.num_blocks,
            self.allow_color_repeat,
            self.color_pool,
            self.near_margin,
            self.block_size_xy,
            self.block_height,
            self.agent_center_start,
            block_torus_wrap=block_torus_wrap,
            box_tex=box_tex,
            box_tex_randomize=box_tex_randomize,
            wall_tex_randomize=wall_tex_randomize,
            floor_tex_randomize=floor_tex_randomize,
            box_and_ball=box_and_ball,
            **kwargs,
        )

    def _gen_world(self):
        # Create a rectangular room
        room_kwargs = {}
        # Floor texture choice (randomization overrides explicit override)
        if self.floor_tex_randomize:
            room_kwargs["floor_tex"] = str(self.np_random.choice(self._floor_tex_pool))
        elif getattr(self, "_floor_tex_override", None) is not None:
            room_kwargs["floor_tex"] = self._floor_tex_override
        # Wall texture choice (randomization overrides explicit override)
        if self.wall_tex_randomize:
            room_kwargs["wall_tex"] = str(self.np_random.choice(self._wall_tex_pool))
        elif getattr(self, "_wall_tex_override", None) is not None:
            room_kwargs["wall_tex"] = self._wall_tex_override
        # Ceiling override unchanged
        if getattr(self, "_ceil_tex_override", None) is not None:
            room_kwargs["ceil_tex"] = self._ceil_tex_override
        # wall_tex="concrete",
        # ceil_tex="concrete_tiles",
        self.add_rect_room(min_x=0, max_x=self.size, min_z=0, max_z=self.size, **room_kwargs)

        # Choose colors for the blocks
        if self.ensure_base_palette and self.num_blocks > 0:
            base_palette = ["green", "red", "yellow", "blue", "purple", "grey"]
            # Filter base colors to those present in the pool
            base_palette = [c for c in base_palette if c in self.color_pool]
            k = min(self.num_blocks, len(base_palette))
            # Ensure at least one of each (up to k)
            chosen_colors = list(self.np_random.choice(base_palette, size=k, replace=False))
            # Fill the remainder randomly from the full pool (with replacement)
            rem = self.num_blocks - k
            if rem > 0:
                extra = list(self.np_random.choice(self.color_pool, size=rem, replace=True))
                chosen_colors.extend(extra)
        else:
            if not self.allow_color_repeat and self.num_blocks <= len(self.color_pool):
                chosen_colors = list(self.np_random.choice(self.color_pool, size=self.num_blocks, replace=False))
            else:
                chosen_colors = list(self.np_random.choice(self.color_pool, size=self.num_blocks, replace=True))

        # Helper to compute room extents with optional spawn buffer
        def _spawn_extents(ent_radius: float):
            if self.spawn_wall_buffer is None:
                return None, None, None, None
            buf = float(self.spawn_wall_buffer) + float(ent_radius)
            min_x = 0.0 + buf
            max_x = self.size + buf - 1.0
            min_z = 0.0 + buf
            max_z = self.size + buf - 1.0 # Plus buf because the buf is padded from the front # Minus 1 because the room size in grid mode is 1 less than actual.
            # Ensure valid ranges
            if max_x <= min_x or max_z <= min_z:
                return None, None, None, None
            return min_x, max_x, min_z, max_z

        # Place the agent before blocks: either centered or random within buffer
        if self.agent_center_start:
            # Compute the center of the spawnable area considering wall buffer
            mnx, mxx, mnz, mxz = _spawn_extents(self.agent.radius if hasattr(self.agent, 'radius') else 0.5)
            if mnx is None:
                # No buffer; full room extents
                mnx, mxx, mnz, mxz = 0.0, float(self.size), 0.0, float(self.size)
            cx = (mnx + mxx) / 2.0
            cz = (mnz + mxz) / 2.0
            
            # Adjust snapping per requested convention
            if self.grid_mode:
                # Favor top-left if midpoint is exactly an integer; else floor
                if abs(cx - round(cx)) < 1e-6:
                    cx = round(cx) - 1.0
                else:
                    cx = math.floor(cx)
                if abs(cz - round(cz)) < 1e-6:
                    cz = round(cz) - 1.0
                else:
                    cz = math.floor(cz)
            else:
                # If exactly centered on an integer, offset by 0.5 toward top-left
                if abs(cx - round(cx)) < 1e-6:
                    cx = cx - 0.5
                if abs(cz - round(cz)) < 1e-6:
                    cz = cz - 0.5
            # Clamp to buffered extents for safety
            cx = min(max(cx, mnx), mxx)
            cz = min(max(cz, mnz), mxz)
            center_pos = np.array([float(cx), 0.0, float(cz)], dtype=float)
            self.place_agent(pos=center_pos)
        else:
            mnx, mxx, mnz, mxz = _spawn_extents(self.agent.radius if hasattr(self.agent, 'radius') else 0.5)
            self.place_agent(min_x=mnx, max_x=mxx, min_z=mnz, max_z=mxz)

        # Place blocks with either uniform or random sizes
        for color in chosen_colors:
            # Choose texture per box if randomization is enabled; else use override
            if self.box_tex_randomize:
                tex_name = str(self.np_random.choice(self._box_tex_pool))
            else:
                tex_name = self._box_tex_override
            # Choose entity type if enabled, 1 in 3 chance to be a ball
            spawn_ball = bool(self.box_and_ball and (int(self.np_random.integers(0, 3)) == 1))
            if spawn_ball:
                # ent = Ball(color=color, size=float(self.np_random.uniform(0.6, 0.85)))
                ent = Ball(color=color, size=0.85)
            else:
                if (self.block_size_xy is not None) or (self.block_height is not None):
                    # Use provided controls; default footprint if only height provided
                    sx = self.block_size_xy if self.block_size_xy is not None else 0.75
                    sy = self.block_height if self.block_height is not None else sx
                    sz = sx
                    ent = Box(color=color, size=np.array([sx, sy, sz], dtype=float), texture=tex_name)
                else:
                    # Backward-compatible random cube size
                    ent = Box(color=color, size=self.np_random.uniform(0.6, 0.85), texture=tex_name)
            # Re-try placement if spawned exactly at the agent's (x,z)
            attempts = 0
            while True:
                attempts += 1
                mnx, mxx, mnz, mxz = _spawn_extents(ent.radius if hasattr(ent, 'radius') else 0.3)
                self.place_entity(
                    ent,
                    min_x=mnx,
                    max_x=mxx,
                    min_z=mnz,
                    max_z=mxz,
                )
                # Compare only XZ for exact location match
                if hasattr(self, 'agent') and (self.agent is not None):
                    same_x = abs(float(ent.pos[0]) - float(self.agent.pos[0])) < 1e-6
                    same_z = abs(float(ent.pos[2]) - float(self.agent.pos[2])) < 1e-6
                    if same_x and same_z:
                        # Remove and retry
                        try:
                            self.entities.remove(ent)
                        except ValueError:
                            pass
                        if attempts < 100:
                            continue
                break

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
                if isinstance(ent, (Box, Ball)):
                    ent.dir = 0.0

        if self.grid_mode:
            self._snap_entity_to_grid(self.agent)
            for ent in self.entities:
                if isinstance(ent, (Box, Ball)):
                    self._snap_entity_to_grid(ent)

        # Assign velocities
        rand = self.np_random if self.domain_rand else None
        move_step = self.params.sample(rand, "forward_step") * self.box_speed_scale

        for ent in self.entities:
            if not isinstance(ent, (Box, Ball)):
                continue
            if self.blocks_static:
                # Do not assign velocities for static mode
                ent.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
                continue
            if self.grid_mode:
                if self.grid_cardinal_only:
                    # Choose one of 4 cardinal directions uniformly
                    choice = int(self.np_random.integers(0, 4))
                    speed = max(1, max(abs(self.grid_vel_min), abs(self.grid_vel_max)))
                    if choice == 0:
                        vx, vz = speed, 0
                    elif choice == 1:
                        vx, vz = -speed, 0
                    elif choice == 2:
                        vx, vz = 0, speed
                    else:
                        vx, vz = 0, -speed
                else:
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
            if self.box_allow_overlap and isinstance(ent, (Box, Ball)) and isinstance(ent2, (Box, Ball)):
                continue
            if self.agent_box_allow_overlap and (
                (ent is self.agent and isinstance(ent2, (Box, Ball)))
                or (ent2 is self.agent and isinstance(ent, (Box, Ball)))
            ):
                continue

            px, _, pz = ent2.pos
            pos2 = np.array([px, 0, pz])
            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def _intersect_entities_only(self, ent, pos, radius):
        # Check only entity collisions (ignore walls). Returns colliding entity or None.
        px, _, pz = pos
        pos = np.array([px, 0, pz])

        for ent2 in self.entities:
            if ent2 is ent:
                continue
            if self.box_allow_overlap and isinstance(ent, (Box, Ball)) and isinstance(ent2, (Box, Ball)):
                continue
            if self.agent_box_allow_overlap and (
                (ent is self.agent and isinstance(ent2, (Box, Ball)))
                or (ent2 is self.agent and isinstance(ent, (Box, Ball)))
            ):
                continue

            px2, _, pz2 = ent2.pos
            pos2 = np.array([px2, 0, pz2])
            d = np.linalg.norm(pos2 - pos)
            if d < radius + ent2.radius:
                return ent2

        return None

    def _wrap_coord_inside(self, value: float, radius: float) -> float:
        """Wrap value onto [radius, size - radius], inclusive of radius and exclusive of (size - radius) upper bound.
        If the interior width is non-positive, clamp instead.
        """
        lo = float(0.0 + radius)
        hi = float(self.size - radius)
        interior = hi - lo
        if interior <= 0.0:
            # Degenerate case: just clamp
            return min(max(value, lo), hi)
        return (value - lo) % interior + lo

    def _wrap_across_axis(self, center_val: float, radius: float) -> float:
        """Wrap a circle center across 0..size boundaries so it appears on the opposite side
        strictly inside the room by at least its radius.
        """
        if center_val + radius > self.size:
            new_val = center_val - self.size
            return max(new_val, radius)
        if center_val - radius < 0.0:
            new_val = center_val + self.size
            return min(new_val, self.size - radius)
        return center_val

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
            if not isinstance(ent, (Box, Ball)):
                continue
            if carrying is not None and ent is carrying:
                continue
            if self.blocks_static:
                # No motion updates for static mode
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
                if self.block_torus_wrap:
                    # Unconditionally wrap across boundaries; ignore wall collisions
                    if (candidate[0] + ent.radius > self.size) or (candidate[0] - ent.radius < 0.0):
                        wrapped = current_pos.copy()
                        wrapped[0] = self._wrap_across_axis(candidate[0], ent.radius)
                        current_pos = wrapped
                    else:
                        # Move unless blocked by entity; if blocked, bounce off entity
                        hit_ent = self._intersect_entities_only(ent, candidate, ent.radius)
                        if (not hit_ent) or (self.box_allow_overlap and isinstance(hit_ent, (Box, Ball))):
                            current_pos = candidate
                        else:
                            ent.velocity[0] = -ent.velocity[0]
                            candidate_bounce = current_pos.copy()
                            candidate_bounce[0] += ent.velocity[0]
                            hit_bounce = self._intersect_entities_only(ent, candidate_bounce, ent.radius)
                            if (not hit_bounce) or (self.box_allow_overlap and isinstance(hit_bounce, (Box, Ball))):
                                current_pos = candidate_bounce
                else:
                    hit = self.intersect(ent, candidate, ent.radius)
                    if (not hit) or (self.box_allow_overlap and isinstance(hit, (Box, Ball))):
                        current_pos = candidate
                    else:
                        ent.velocity[0] = -ent.velocity[0]
                        candidate_bounce = current_pos.copy()
                        candidate_bounce[0] += ent.velocity[0]
                        hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                        if (not hit_bounce) or (
                            self.box_allow_overlap and isinstance(hit_bounce, (Box, Ball))
                        ):
                            current_pos = candidate_bounce
            if self.grid_mode:
                if self.block_torus_wrap:
                    # Keep blocks off the wall cells when wrapping in grid mode
                    snapped = round(current_pos[0])
                    snapped = min(max(snapped, 1.0), float(self.size - 1))
                    current_pos[0] = snapped
                else:
                    current_pos[0] = round(current_pos[0])

            # Z axis
            if ent.velocity[2] != 0.0:
                candidate = current_pos.copy()
                candidate[2] += ent.velocity[2]
                if self.block_torus_wrap:
                    if (candidate[2] + ent.radius > self.size) or (candidate[2] - ent.radius < 0.0):
                        wrapped = current_pos.copy()
                        wrapped[2] = self._wrap_across_axis(candidate[2], ent.radius)
                        current_pos = wrapped
                    else:
                        hit_ent = self._intersect_entities_only(ent, candidate, ent.radius)
                        if (not hit_ent) or (self.box_allow_overlap and isinstance(hit_ent, (Box, Ball))):
                            current_pos = candidate
                        else:
                            ent.velocity[2] = -ent.velocity[2]
                            candidate_bounce = current_pos.copy()
                            candidate_bounce[2] += ent.velocity[2]
                            hit_bounce = self._intersect_entities_only(ent, candidate_bounce, ent.radius)
                            if (not hit_bounce) or (self.box_allow_overlap and isinstance(hit_bounce, (Box, Ball))):
                                current_pos = candidate_bounce
                else:
                    hit = self.intersect(ent, candidate, ent.radius)
                    if (not hit) or (self.box_allow_overlap and isinstance(hit, (Box, Ball))):
                        current_pos = candidate
                    else:
                        ent.velocity[2] = -ent.velocity[2]
                        candidate_bounce = current_pos.copy()
                        candidate_bounce[2] += ent.velocity[2]
                        hit_bounce = self.intersect(ent, candidate_bounce, ent.radius)
                        if (not hit_bounce) or (
                            self.box_allow_overlap and isinstance(hit_bounce, (Box, Ball))
                        ):
                            current_pos = candidate_bounce
            if self.grid_mode:
                if self.block_torus_wrap:
                    snapped = round(current_pos[2])
                    snapped = min(max(snapped, 1.0), float(self.size - 1))
                    current_pos[2] = snapped
                else:
                    current_pos[2] = round(current_pos[2])

            ent.pos = current_pos

        # No reward-based termination condition here
        return obs, reward, termination, truncation, info


