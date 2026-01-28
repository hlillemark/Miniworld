#!/usr/bin/env python3

"""
scripts.generate_videos

Generate MiniWorld rollout videos (RGB + depth) plus action/trajectory metadata for a simple
agent policy. Intended for quick single-rollout generation and debugging.

Basic usage:
  python -m scripts.generate_videos \
    --env-name MiniWorld-MovingBlockWorld-v0 \
    --steps 300 --out-prefix ./out/run1

Outputs (written using <out-prefix>):
  - <out-prefix>_rgb.mp4          RGB rollout video (T frames)
  - <out-prefix>_depth.pt         Raw depth tensor float32, shape (T,H,W,1)
  - <out-prefix>_actions.pt       Dict with:
        actions (T,), agent_pos (T,3), delta_xz (T,2), delta_dir (T,), agent_dir (T,)

Optional debug/map outputs:
  - --debug-join                  Saves <out-prefix>_debug.mp4 (RGB | top-view side-by-side)
  - --output-2d-map               Saves <out-prefix>_map_2d.mp4 and stores top_view_scale in actions.pt

Key flag groups:
  Rendering:
    --render-width/--render-height     RGB video resolution (env.render())
    --obs-width/--obs-height           Depth/obs buffer resolution

  Episode / environment:
    --room-size, --spawn-wall-buffer, --no-time-limit, --domain-rand, --heading-zero,
    --agent-center-start

  Discrete motion / camera:
    --turn-step-deg, --forward-step, --cam-fov-y, --even-lighting

  Blocks / dynamics (MovingBlockWorld):
    --grid-mode + --grid-vel-min/max [--grid-cardinal-only], --blocks-static, --num-blocks,
    --num-blocks-min/max, --ensure-base-palette, --block-size-xy, --block-height,
    overlap/orientation flags (e.g., --box-allow-overlap, --agent-box-allow-overlap, --box-random-orientation)

  Textures:
    --floor-tex, --wall-tex, --ceil-tex, --box-tex, plus randomizers:
    --randomize-(wall|floor|box)-tex, --box-and-ball

Policies (--policy):
  biased_random (default): forward-biased random walk with wall avoidance knobs:
    --forward-prob, --turn-left-weight, --turn-right-weight, --wall-buffer,
    --avoid-turning-into-walls, --lookahead-mult
  biased_walk_v2: wall-biased exploration with periodic "look" toward center (uses --forward-prob, --observe-steps)
  back_and_forth: straight segments then turn around (uses --segment-len)
  center_rotate: random left/right/NOOP (NOOP encoded as action id 4)
  do_nothing: always NOOP (action id 4)
  edge_plus: cycles between edge centers and pauses to observe (uses --observe-steps)
  peeakboo / peekaboo_motion: inward/outward observation cycles (uses --observe-steps / --observe-inward-steps / --observe-outward-steps)
  blockmover: pick up and relocate blocks (forces collisions on)

Headless rendering (servers):
  Set MINIWORLD_HEADLESS=1 to force pyglet + software GL headless mode:
    MINIWORLD_HEADLESS=1 python -m scripts.generate_videos ...

Notes:
  Dataset/multi-generation flags (--dataset-root, --num-videos, etc.) are DEPRECATED in this
  script; use scripts.generate_videos_batch for large-scale generation.

Command to generate one sample from the textured training set used in the FloWM paper: 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --out-prefix ./out/tex --debug-join \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball
  
"""


import argparse
import math
import os
from typing import Tuple, Optional

import gymnasium as gym
import numpy as np
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from types import SimpleNamespace
from pathlib import Path


def _is_truthy(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _should_enable_headless() -> bool:
    # Explicit opt-in via env var takes precedence
    if _is_truthy(os.environ.get("MINIWORLD_HEADLESS", "")):
        return True
    # Respect existing pyglet hint if already set by caller
    if _is_truthy(os.environ.get("PYGLET_HEADLESS", "")):
        return True
    # Fall back to auto-detect: no X/Wayland display implies headless
    if (os.name != "nt") and (not os.environ.get("DISPLAY")) and (not os.environ.get("WAYLAND_DISPLAY")):
        return True
    return False

_HEADLESS_MODE = _should_enable_headless()

if _HEADLESS_MODE:
    os.environ.setdefault("PYGLET_HEADLESS", "1")
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "llvmpipe")
    os.environ.setdefault("MESA_GL_VERSION_OVERRIDE", "3.3")
    os.environ.setdefault("MESA_GLSL_VERSION_OVERRIDE", "330")

import multiprocessing as mp
if mp.get_start_method(allow_none=True) != "spawn":
    mp.set_start_method("spawn", force=True)

# optional: avoid hidden X/GLX “shadow window” (only when headless)
import pyglet
if _HEADLESS_MODE:
    pyglet.options['shadow_window'] = False

import miniworld
from miniworld.params import DEFAULT_PARAMS
from miniworld.entity import Box


def build_env(args) -> gym.Env:
    view_mode = "top" if args.top_view else "agent"
    env_kwargs = {
        "view": view_mode,
        "render_mode": "rgb_array",
        # Observation buffer size (used by render_obs and default depth)
        "obs_width": int(args.obs_width),
        "obs_height": int(args.obs_height),
        # Human/render buffer size (controls env.render() RGB video resolution)
        "window_width": int(args.render_width),
        "window_height": int(args.render_height),
    }

    # Room size if provided
    if args.room_size is not None:
        env_kwargs["size"] = int(args.room_size)

    # Floor texture override if provided
    if getattr(args, "floor_tex", None):
        env_kwargs["floor_tex"] = str(args.floor_tex)
    # Wall/Ceiling texture override if provided
    if getattr(args, "wall_tex", None):
        env_kwargs["wall_tex"] = str(args.wall_tex)
    if getattr(args, "ceil_tex", None):
        env_kwargs["ceil_tex"] = str(args.ceil_tex)
    # Box texture override if provided
    if getattr(args, "box_tex", None):
        env_kwargs["box_tex"] = str(args.box_tex)
    # Per-box texture randomization (overrides box_tex if set)
    if getattr(args, "randomize_box_tex", False):
        env_kwargs["box_tex_randomize"] = True
    # Wall/Floor randomization flags (override respective specific tex if set)
    if getattr(args, "randomize_wall_tex", False):
        env_kwargs["wall_tex_randomize"] = True
    if getattr(args, "randomize_floor_tex", False):
        env_kwargs["floor_tex_randomize"] = True
    # Randomize spawning boxes/balls
    if getattr(args, "box_and_ball", False):
        env_kwargs["box_and_ball"] = True

    # Map supported custom kwargs
    if args.box_speed_scale is not None:
        env_kwargs["box_speed_scale"] = args.box_speed_scale
    if args.box_allow_overlap:
        env_kwargs["box_allow_overlap"] = True
    if args.agent_box_allow_overlap:
        env_kwargs["agent_box_allow_overlap"] = True
    if args.box_random_orientation:
        env_kwargs["box_random_orientation"] = True
    if args.grid_mode:
        env_kwargs["grid_mode"] = True
        env_kwargs["grid_vel_min"] = int(args.grid_vel_min)
        env_kwargs["grid_vel_max"] = int(args.grid_vel_max)
        if getattr(args, "grid_cardinal_only", False):
            env_kwargs["grid_cardinal_only"] = True
    # Block count stochasticity and color palette controls (MovingBlockWorld)
    if getattr(args, "num_blocks", None) is not None:
        env_kwargs["num_blocks"] = int(args.num_blocks)
    if getattr(args, "num_blocks_min", None) is not None and getattr(args, "num_blocks_max", None) is not None:
        # Defer sampling to env.reset by passing a random value each build
        import random as _py_random
        env_kwargs["num_blocks"] = int(_py_random.randint(int(args.num_blocks_min), int(args.num_blocks_max)))
    if getattr(args, "ensure_base_palette", False):
        env_kwargs["ensure_base_palette"] = True
    if getattr(args, "blocks_static", False):
        env_kwargs["blocks_static"] = True
    if getattr(args, "block_torus_wrap", False):
        env_kwargs["block_torus_wrap"] = True
    if getattr(args, "spawn_wall_buffer", None) is not None:
        env_kwargs["spawn_wall_buffer"] = float(args.spawn_wall_buffer)
    # Uniform block sizing controls (MovingBlockWorld)
    if getattr(args, "block_size_xy", None) is not None:
        env_kwargs["block_size_xy"] = float(args.block_size_xy)
    if getattr(args, "block_height", None) is not None:
        env_kwargs["block_height"] = float(args.block_height)
    # Agent spawn control: center start
    if getattr(args, "agent_center_start", False):
        env_kwargs["agent_center_start"] = True

    # Build params so first reset uses them
    params = None
    if args.turn_step_deg is not None or args.forward_step is not None:
        params = DEFAULT_PARAMS.no_random()
        if args.turn_step_deg is not None:
            v = float(args.turn_step_deg)
            params.set("turn_step", v, v, v)
        if args.forward_step is not None:
            v = float(args.forward_step)
            params.set("forward_step", v, v, v)
    # Camera vertical field of view (degrees)
    if getattr(args, "cam_fov_y", None) is not None:
        if params is None:
            params = DEFAULT_PARAMS.copy()
        v = float(args.cam_fov_y)
        params.set("cam_fov_y", v, v, v)
    # Even lighting: force ambient 1, diffuse 0 regardless of domain rand
    if getattr(args, "even_lighting", False):
        if params is None:
            params = DEFAULT_PARAMS.copy()
        params.set("light_ambient", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        params.set("light_color", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    if params is not None:
        env_kwargs["params"] = params

    env = gym.make(args.env_name, **env_kwargs)

    # Domain randomization and time limit controls (like manual_control)
    env_unwrapped = env.unwrapped
    if args.no_time_limit:
        env_unwrapped.max_episode_steps = np.inf
    if args.domain_rand:
        env_unwrapped.domain_rand = True
    if args.heading_zero:
        # Will be applied after first reset
        pass
    return env


class BackAndForthPolicy:
    """Move forward in a straight line for segment_len steps, then turn around (left,left), repeat."""

    def __init__(self, segment_len: int = 40):
        assert segment_len >= 1
        self.segment_len = int(segment_len)
        self.cycle_len = self.segment_len + 2

    def action(self, step_idx: int) -> int:
        pos = step_idx % self.cycle_len
        if pos < self.segment_len:
            return 2  # move_forward
        elif pos == self.segment_len or pos == self.segment_len + 1:
            return 0  # turn_left
        return 2


class BiasedRandomPolicy:
    """Biased random exploration that prefers moving forward, avoids walls, and turns when needed.

    Knobs:
      - forward_prob: probability of attempting to move forward when safe
      - turn_left_weight / turn_right_weight: relative weights when choosing a turn
      - wall_buffer: keep at least this many meters from walls; turn away when too close
      - avoid_turning_into_walls: if True, when turning, prefer direction that increases distance to walls
    """

    def __init__(
        self,
        env: gym.Env,
        forward_prob: float = 0.8,
        turn_left_weight: float = 1.0,
        turn_right_weight: float = 1.0,
        wall_buffer: float = 1.5,
        avoid_turning_into_walls: bool = True,
        lookahead_mult: float = 2.0,
    ):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.forward_prob = float(forward_prob)
        self.turn_left_weight = float(turn_left_weight)
        self.turn_right_weight = float(turn_right_weight)
        self.wall_buffer = float(wall_buffer)
        self.avoid_turning_into_walls = bool(avoid_turning_into_walls)
        self.lookahead_mult = float(lookahead_mult)
        # Stuck mitigation state
        self._last_pos = None
        self._prev_action = None
        self._turn_streak = 0
        self._prefer_forward_next = False  # after a turn, attempt one forward if safe
        self._last_turn_dir = 0  # +1 for left, -1 for right, 0 none

    def _dist_to_walls(self, pos: np.ndarray) -> float:
        x, _, z = pos
        return min(
            x - self.env.min_x,
            self.env.max_x - x,
            z - self.env.min_z,
            self.env.max_z - z,
        )

    def _ahead_pos(self, pos: np.ndarray, dir_rad: float, dist: float) -> np.ndarray:
        dx = math.cos(dir_rad) * dist
        dz = -math.sin(dir_rad) * dist
        nxt = pos.copy()
        nxt[0] += dx
        nxt[2] += dz
        return nxt

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent
        fwd_step = float(self.env.max_forward_step)
        lookahead = fwd_step * self.lookahead_mult

        # Forward candidate (one step)
        next_pos = self._ahead_pos(agent.pos, agent.dir, fwd_step)
        forward_collides = bool(self.env.intersect(agent, next_pos, agent.radius))

        def turn_dir_score(turn_sign: int) -> float:
            # turn_sign: +1 left, -1 right
            turn_step_deg = self.env.params.get_max("turn_step")
            turn_step_rad = float(turn_step_deg) * math.pi / 180.0
            new_dir = agent.dir + (turn_step_rad if turn_sign > 0 else -turn_step_rad)
            ahead2 = self._ahead_pos(agent.pos, new_dir, lookahead)
            return self._dist_to_walls(ahead2)

        # If forward would collide, choose a turn (prefer direction with more clearance)
        if forward_collides:
            left_score = turn_dir_score(+1)
            right_score = turn_dir_score(-1)
            if self.avoid_turning_into_walls and (left_score != right_score):
                return a.turn_left if left_score > right_score else a.turn_right
            probs = np.array([self.turn_left_weight, self.turn_right_weight], dtype=float)
            probs = probs / probs.sum()
            return a.turn_left if self.rng.random() < probs[0] else a.turn_right

        # Otherwise: move forward with probability, else turn (optionally away from walls)
        if self.rng.random() < self.forward_prob:
            return a.move_forward

        if self.avoid_turning_into_walls:
            left_score = turn_dir_score(+1)
            right_score = turn_dir_score(-1)
            if left_score != right_score:
                return a.turn_left if left_score > right_score else a.turn_right

        probs = np.array([self.turn_left_weight, self.turn_right_weight], dtype=float)
        probs = probs / probs.sum()
        return a.turn_left if self.rng.random() < probs[0] else a.turn_right


class CenterRotatePolicy:
    """
    Rotation experiment policy.
    Each step choose uniformly among {turn_left(0), turn_right(1), NOOP(4)}.

    Dataset convention: action id 4 is used as NOOP. In-env, id 4 is "pickup";
    set --forward-step 0 to make it have no effect.
    """

    def __init__(self, env: gym.Env):
        self.env = env.unwrapped
        self.rng = self.env.np_random

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        r = int(self.rng.integers(0, 3))  # 0,1,2 -> left,right,noop
        if r == 0:
            return a.turn_left
        if r == 1:
            return a.turn_right
        return a.do_nothing  # id 4 used as NOOP in datasets
    

class DoNothingPolicy:
    """
    action id 4 is used as noop
    Does nothing, just stands there.
    """

    def __init__(self, env: gym.Env):
        self.env = env.unwrapped

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        return a.do_nothing


class BlockMoverPolicy:
    """
    Block mover with clear phases:
      1) select_block -> choose the block to move
      2) plan_to_block -> compute axis-aligned approach pose (same x or z, facing the block)
      3) move_to_block -> execute the plan (turn/forward only)
      4) pickup -> issue pickup
      5) plan_to_target -> pick empty target and compute agent pose that will drop block in front
      6) move_to_target -> execute plan
      7) drop -> issue drop, then repeat
    """

    def __init__(self, env: gym.Env):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        # Ensure collisions are enforced (needed for pickup/drop/intersections)
        if hasattr(self.env, "agent_box_allow_overlap"):
            self.env.agent_box_allow_overlap = False
        if hasattr(self.env, "box_allow_overlap"):
            self.env.box_allow_overlap = False
        # Cached step sizes
        self.turn_step_rad = float(self.env.params.get_max("turn_step")) * math.pi / 180.0
        self.fwd_step = float(self.env.max_forward_step)
        # State
        self.phase = "select_block"
        self.block: Optional[Box] = None
        self.target_pos: Optional[tuple] = None  # (x,z) for drop
        self.plan_actions: list[int] = []
        self.approach_pose: Optional[tuple] = None  # (x,z,dir_rad)

    # -------- helpers --------
    def _quantize(self, dir_rad: float) -> float:
        s = self.turn_step_rad
        return (round((dir_rad % (2 * math.pi)) / s) * s) % (2 * math.pi)

    def _turn_steps_to(self, desired: float) -> list:
        a = self.env.actions
        curr = float(self.env.agent.dir)
        step = self.turn_step_rad
        c = self._quantize(curr)
        d = self._quantize(desired)
        diff = (d - c + math.pi) % (2 * math.pi) - math.pi
        n = int(round(abs(diff) / step))
        if n == 0:
            return []
        act = a.turn_left if diff > 0 else a.turn_right
        return [act] * n

    def _forward_safe_from(self, x: float, z: float, dir_rad: float, carrying: Optional[Box]) -> bool:
        agent = self.env.agent
        nx = x + math.cos(dir_rad) * self.fwd_step
        nz = z - math.sin(dir_rad) * self.fwd_step
        pos = agent.pos.copy()
        d = float(agent.dir)
        # simulate next pose
        agent.pos[0] = nx
        agent.pos[2] = nz
        agent.dir = dir_rad
        blocked = bool(self.env.intersect(agent, agent.pos, agent.radius))
        if not blocked and carrying is not None:
            cpos = self.env._get_carry_pos(agent.pos, carrying)
            blocked = bool(self.env.intersect(carrying, cpos, carrying.radius))
        # restore
        agent.pos[:] = pos
        agent.dir = d
        return not blocked

    def _build_axis_plan_from_current(self, tx: float, tz: float, tdir: float, carrying: Optional[Box]) -> Optional[list]:
        a = self.env.actions
        plan: list = []
        # local sim
        ax = float(self.env.agent.pos[0])
        az = float(self.env.agent.pos[2])
        ad = float(self.env.agent.dir)
        eps = max(0.25 * self.fwd_step, 1e-3)

        def sim_turns(des: float, p: list, x: float, z: float, d: float) -> tuple:
            turns = self._turn_steps_to(des)
            for t in turns:
                p.append(t)
                d = (d + (self.turn_step_rad if t == a.turn_left else -self.turn_step_rad)) % (2 * math.pi)
            return x, z, d

        def sim_forward_many(dist: float, heading: float, p: list, x: float, z: float, d: float) -> tuple:
            steps = int(math.ceil(dist / self.fwd_step))
            for _ in range(max(0, steps)):
                if not self._forward_safe_from(x, z, heading, carrying):
                    return None
                p.append(a.move_forward)
                x += math.cos(heading) * self.fwd_step
                z -= math.sin(heading) * self.fwd_step
            return (x, z, d)

        # Try two orders: X then Z, or Z then X
        for order in [(True, False), (False, True)]:
            p = []
            x, z, d = ax, az, ad
            ok = True
            if order[0]:
                # move along X
                dir_x = 0.0 if (tx - x) >= 0 else math.pi
                x, z, d = sim_turns(dir_x, p, x, z, d)
                res = sim_forward_many(abs(tx - x), d, p, x, z, d)
                if res is None:
                    ok = False
                else:
                    x, z, d = res
            if ok and order[1]:
                # move along Z
                dir_z = math.pi / 2.0 if (tz - z) < 0 else -math.pi / 2.0
                x, z, d = sim_turns(dir_z, p, x, z, d)
                res = sim_forward_many(abs(tz - z), d, p, x, z, d)
                if res is None:
                    ok = False
                else:
                    x, z, d = res
            if ok:
                # final face
                x, z, d = sim_turns(tdir, p, x, z, d)
                return p
        return None

    def _bfs_plan_to_goal(self, is_goal_fn, carrying: Optional[Box]) -> Optional[list]:
        """Grid BFS over (gx,gz,heading_idx) with actions {turn_left, turn_right, move_forward}."""
        a = self.env.actions
        step = self.fwd_step
        turn_step = self.turn_step_rad
        # Quantizers
        def qpos(x, x0): return int(round((x - x0) / step))
        def qdir(d): return int(round((d % (2 * math.pi)) / turn_step)) % max(1, int(round(2 * math.pi / turn_step)))
        # Start
        sx = float(self.env.agent.pos[0]); sz = float(self.env.agent.pos[2]); sd = float(self.env.agent.dir)
        x0 = float(self.env.min_x); z0 = float(self.env.min_z)
        s_key = (qpos(sx, x0), qpos(sz, z0), qdir(sd))
        from collections import deque
        Q = deque([s_key])
        parent = {s_key: None}
        parent_act = {}
        nodes = 0
        max_nodes = 20000
        # Helper to reconstruct plan
        def reconstruct(key):
            seq = []
            cur = key
            while parent[cur] is not None:
                seq.append(parent_act[cur])
                cur = parent[cur]
            seq.reverse()
            return seq
        while Q:
            key = Q.popleft()
            nodes += 1
            if nodes > max_nodes:
                break
            gx, gz, hk = key
            # Map back to pose
            x = x0 + gx * step
            z = z0 + gz * step
            d = hk * turn_step
            # Goal test in continuous coords
            if is_goal_fn(x, z, d):
                return reconstruct(key)
            # Neighbors: turns
            for act, nhk in ((a.turn_left, (hk + 1) % int(round(2 * math.pi / turn_step))),
                             (a.turn_right, (hk - 1) % int(round(2 * math.pi / turn_step)))):
                nkey = (gx, gz, nhk)
                if nkey in parent:
                    continue
                parent[nkey] = key
                parent_act[nkey] = act
                Q.append(nkey)
            # Forward
            nd = d
            nx = x + math.cos(nd) * step
            nz = z - math.sin(nd) * step
            # Bounds
            if not (self.env.min_x <= nx <= self.env.max_x and self.env.min_z <= nz <= self.env.max_z):
                continue
            # Collision
            if not self._forward_safe_from(x, z, nd, carrying):
                continue
            nkey = (qpos(nx, x0), qpos(nz, z0), qdir(nd))
            if nkey in parent:
                continue
            parent[nkey] = key
            parent_act[nkey] = a.move_forward
            Q.append(nkey)
        return None

    def _choose_block(self) -> Optional[Box]:
        blocks = [e for e in self.env.entities if isinstance(e, Box)]
        if not blocks:
            return None
        return blocks[0] if len(blocks) == 1 else blocks[int(self.rng.integers(0, len(blocks)))]

    def _choose_drop_target(self, carrying: Box) -> Optional[tuple]:
        # Sample random empty locations
        min_x = float(self.env.min_x + 1.0)
        max_x = float(self.env.max_x - 1.0)
        min_z = float(self.env.min_z + 1.0)
        max_z = float(self.env.max_z - 1.0)
        for _ in range(64):
            x = float(self.rng.uniform(min_x, max_x))
            z = float(self.rng.uniform(min_z, max_z))
            # Check block footprint free
            pos = np.array([x, 0.0, z], dtype=float)
            if self.env.intersect(carrying, pos, carrying.radius):
                continue
            return (x, z)
        return None

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent

        if self.phase == "select_block":
            self.block = self._choose_block()
            if self.block is None:
                return a.pickup
            self.phase = "plan_to_block"
            return a.pickup

        if self.phase == "plan_to_block":
            b = self.block
            assert b is not None
            standoff = float(agent.radius + b.radius + 0.25 * self.fwd_step)
            bx = float(b.pos[0]); bz = float(b.pos[2])
            candidates = [
                (bx - standoff, bz, 0.0),                 # West of block, face +X
                (bx + standoff, bz, math.pi),             # East, face -X
                (bx, bz - standoff, math.pi / 2.0),       # North, face +Z
                (bx, bz + standoff, -math.pi / 2.0),      # South, face -Z
            ]
            for tx, tz, tdir in candidates:
                # Try simple axis plan first
                plan = self._build_axis_plan_from_current(tx, tz, tdir, None)
                if not plan:
                    # Fallback BFS with relaxed goal: same x or z as block (within 0.5*step) and dist <= 1.5, facing toward block
                    def goal_fn(x, z, d):
                        same_x = abs(x - bx) <= 0.5 * self.fwd_step
                        same_z = abs(z - bz) <= 0.5 * self.fwd_step
                        facing_ok = abs(((self._quantize(self._dir_to(bx, bz)) - self._quantize(d)) + math.pi) % (2*math.pi) - math.pi) <= (self.turn_step_rad * 0.5 + 1e-6)
                        dist_ok = math.hypot(bx - x, bz - z) <= 1.5
                        return (same_x or same_z) and facing_ok and dist_ok
                    plan = self._bfs_plan_to_goal(goal_fn, None)
                if plan:
                    self.plan_actions = plan
                    self.approach_pose = (tx, tz, tdir)
                    self.phase = "move_to_block"
                    return a.pickup
            # If none feasible, rotate to change heading then retry next step
            return a.turn_left

        if self.phase == "move_to_block":
            if self.plan_actions:
                return int(self.plan_actions.pop(0))
            # Reached approach pose; pickup next
            self.phase = "pickup"
            return a.pickup

        if self.phase == "pickup":
            self.phase = "plan_to_target"
            return a.pickup

        if self.phase == "plan_to_target":
            carrying = agent.carrying
            if carrying is None:
                # If pickup failed yet, try again
                return a.pickup
            drop = self._choose_drop_target(carrying)
            if drop is None:
                return a.turn_right
            tx, tz = drop
            # Choose a cardinal facing so block drops at (tx,tz)
            headings = [0.0, math.pi/2.0, math.pi, -math.pi/2.0]
            for h in headings:
                # Compute agent pose that will place block in front
                d = float(agent.radius + carrying.radius + self.fwd_step) * 1.05
                ax = tx - math.cos(h) * d
                az = tz + math.sin(h) * d
                plan = self._build_axis_plan_from_current(ax, az, h, carrying)
                if plan:
                    self.plan_actions = plan
                    self.target_pos = (tx, tz)
                    self.phase = "move_to_target"
                    return a.pickup
            return a.turn_right

        if self.phase == "move_to_target":
            if self.plan_actions:
                return int(self.plan_actions.pop(0))
            self.phase = "drop"
            return a.pickup

        if self.phase == "drop":
            # Drop the block in front
            self.phase = "select_block"
            self.block = None
            self.target_pos = None
            return a.drop if agent.carrying is not None else a.pickup

        return a.pickup



class EdgePlusPolicy:
    """
    Cycle between the centers of the four room edges (N, E, S, W).
    At each edge center, face inward toward the room center and pause for
    a fixed number of steps by emitting NOOP (dataset action id 4 -> env.pickup).

    Navigation between edges is a simple turn-then-forward controller that
    moves toward the next edge center, passing through the room center.
    """

    def __init__(self, env: gym.Env, observe_steps: int = 5):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.observe_steps = int(max(0, observe_steps))

        # Room geometry
        self.cx = float((self.env.min_x + self.env.max_x) * 0.5)
        self.cz = float((self.env.min_z + self.env.max_z) * 0.5)
        wall_buf = float(getattr(self.env, "spawn_wall_buffer", 1.0))

        # Edge centers: N (min_z), E (max_x), S (max_z), W (min_x)
        self.edge_points = [
            (self.cx, self.env.min_z + wall_buf),
            (self.env.max_x - wall_buf, self.cz),
            (self.cx, self.env.max_z - wall_buf),
            (self.env.min_x + wall_buf, self.cz),
        ]

        # Controller params
        turn_step_deg = float(self.env.params.get_max("turn_step"))
        self.turn_step_rad = turn_step_deg * math.pi / 180.0
        fwd_step = float(self.env.max_forward_step)
        self.reach_eps = max(0.05, 0.5 * fwd_step)

        # State machine: always route via center, turns only at center/edge
        self.current_idx = self._choose_start_edge_index()
        self.target_idx = None  # chosen randomly when leaving center
        # align_inward -> observe_edge -> align_to_center -> forward_to_center -> align_to_edge -> forward_to_edge
        self.phase = "align_inward"
        self.observe_remaining = self.observe_steps

        # Snap start to the chosen edge center if it's free; keep current heading
        self._try_snap_to_edge(self.current_idx)

    def _wrap(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _dir_to(self, x: float, z: float) -> float:
        # Inverse of ahead mapping: dir -> (dx, dz) = (cos(dir), -sin(dir))
        dx = x - float(self.env.agent.pos[0])
        dz = z - float(self.env.agent.pos[2])
        return math.atan2(-dz, dx)

    def _desired_dir_from_delta(self, dx: float, dz: float) -> float:
        return math.atan2(-dz, dx)

    def _is_pos_free(self, x: float, z: float) -> bool:
        agent = self.env.agent
        pos = agent.pos.copy()
        pos[0] = float(x)
        pos[2] = float(z)
        return not bool(self.env.intersect(agent, pos, agent.radius))

    def _set_agent_pose(self, x: float, z: float, dir_rad: float):
        self.env.agent.pos[0] = float(x)
        self.env.agent.pos[2] = float(z)
        self.env.agent.dir = float(self._wrap(dir_rad))

    def _choose_start_edge_index(self) -> int:
        # Choose nearest edge center to current spawn
        ax, az = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        dists = []
        for i, (x, z) in enumerate(self.edge_points):
            d = (x - ax) ** 2 + (z - az) ** 2
            dists.append((d, i))
        dists.sort()
        return dists[0][1]

    def _try_snap_to_edge(self, idx: int):
        # Try selected edge; if blocked, try others in order of increasing distance
        ax, az = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        candidates = []
        for i, (x, z) in enumerate(self.edge_points):
            d = (x - ax) ** 2 + (z - az) ** 2
            candidates.append((d, i, x, z))
        candidates.sort()
        for _, i, x, z in candidates:
            if self._is_pos_free(x, z):
                # Keep current heading; do not rotate directly
                self._set_agent_pose(x, z, self.env.agent.dir)
                self.current_idx = i
                return
        # If all blocked, keep current spawn; do nothing

    def _desired_inward_dir(self) -> float:
        x, z = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        return math.atan2(-(self.cz - z), (self.cx - x))

    def _next_idx(self) -> int:
        # Visit edges in order N -> E -> S -> W -> ...
        return (self.current_idx + 1) % 4

    def _choose_next_edge_random(self) -> int:
        candidates = [0, 1, 2, 3]
        try:
            candidates.remove(self.current_idx)
        except ValueError:
            pass
        j = int(self.rng.integers(0, len(candidates)))
        return int(candidates[j])

    def _turn_toward(self, desired: float) -> Optional[int]:
        """Single-step heading alignment toward desired; returns turn action or None if aligned."""
        a = self.env.actions
        curr = float(self.env.agent.dir)
        err = abs(self._wrap(desired - curr))
        if err <= self.turn_step_rad * 0.5:
            return None
        left_err = abs(self._wrap(desired - (curr + self.turn_step_rad)))
        right_err = abs(self._wrap(desired - (curr - self.turn_step_rad)))
        return a.turn_left if left_err <= right_err else a.turn_right

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent
        ax, az = float(agent.pos[0]), float(agent.pos[2])

        if self.phase == "align_inward":
            desired = self._desired_inward_dir()
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "observe_edge"
            return a.do_nothing

        if self.phase == "observe_edge":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.do_nothing
            # proceed to center route
            self.phase = "align_to_center"

        if self.phase == "align_to_center":
            desired = self._dir_to(self.cx, self.cz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_center"
            return a.do_nothing

        if self.phase == "forward_to_center":
            dx, dz = (self.cx - ax), (self.cz - az)
            dist = math.hypot(dx, dz)
            if dist <= self.reach_eps:
                # snap to exact center and proceed to align toward next edge
                self._set_agent_pose(self.cx, self.cz, agent.dir)
                self.phase = "align_to_edge"
                return a.do_nothing
            # drive straight; if blocked by dynamic obstacle, wait
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.do_nothing

        if self.phase == "align_to_edge":
            if self.target_idx is None:
                self.target_idx = self._choose_next_edge_random()
            tx, tz = self.edge_points[self.target_idx]
            desired = self._dir_to(tx, tz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_edge"
            return a.do_nothing

        if self.phase == "forward_to_edge":
            tx, tz = self.edge_points[self.target_idx]
            dx, dz = (tx - ax), (tz - az)
            dist = math.hypot(dx, dz)
            if dist <= self.reach_eps:
                # snap to edge, then align inward using discrete turns
                self._set_agent_pose(tx, tz, agent.dir)
                self.current_idx = self.target_idx
                self.target_idx = None  # defer next choice to align_to_edge
                self.phase = "align_inward"
                self.observe_remaining = self.observe_steps
                return a.do_nothing
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.do_nothing

        # fallback
        return a.do_nothing


class PeekabooMotionPolicy:
    """
    Edge-visiting controller like EdgePlus, but at each edge:
      - Align inward, observe for observe_inward_steps
      - Align outward (toward wall), observe for observe_outward_steps
      - Align to center, move to center; align to next edge, move to edge; repeat
    """

    def __init__(self, env: gym.Env, observe_inward_steps: int = 5, observe_outward_steps: int = 20):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.observe_inward_steps = int(max(0, observe_inward_steps))
        self.observe_outward_steps = int(max(0, observe_outward_steps))

        # Room geometry
        self.cx = float((self.env.min_x + self.env.max_x) * 0.5)
        self.cz = float((self.env.min_z + self.env.max_z) * 0.5)
        wall_buf = float(getattr(self.env, "spawn_wall_buffer", 1.0))

        # Edge centers: N (min_z), E (max_x), S (max_z), W (min_x)
        self.edge_points = [
            (self.cx, self.env.min_z + wall_buf),
            (self.env.max_x - wall_buf, self.cz),
            (self.cx, self.env.max_z - wall_buf),
            (self.env.min_x + wall_buf, self.cz),
        ]

        # Controller params
        turn_step_deg = float(self.env.params.get_max("turn_step"))
        self.turn_step_rad = turn_step_deg * math.pi / 180.0
        fwd_step = float(self.env.max_forward_step)
        self.reach_eps = max(0.05, 0.5 * fwd_step)

        # State
        self.current_idx = self._choose_start_edge_index()
        self.target_idx = None
        # align_inward -> observe_inward -> align_outward -> observe_outward -> align_to_center -> forward_to_center -> align_to_edge -> forward_to_edge
        self.phase = "align_inward"
        self.observe_remaining = self.observe_inward_steps
        self._try_snap_to_edge(self.current_idx)

    def _wrap(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _dir_to(self, x: float, z: float) -> float:
        dx = x - float(self.env.agent.pos[0])
        dz = z - float(self.env.agent.pos[2])
        return math.atan2(-dz, dx)

    def _is_pos_free(self, x: float, z: float) -> bool:
        agent = self.env.agent
        pos = agent.pos.copy()
        pos[0] = float(x)
        pos[2] = float(z)
        return not bool(self.env.intersect(agent, pos, agent.radius))

    def _set_agent_pose(self, x: float, z: float, dir_rad: float):
        self.env.agent.pos[0] = float(x)
        self.env.agent.pos[2] = float(z)
        self.env.agent.dir = float(self._wrap(dir_rad))

    def _try_snap_to_edge(self, idx: int):
        ax, az = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        candidates = []
        for i, (x, z) in enumerate(self.edge_points):
            d = (x - ax) ** 2 + (z - az) ** 2
            candidates.append((d, i, x, z))
        candidates.sort()
        for _, i, x, z in candidates:
            if self._is_pos_free(x, z):
                self._set_agent_pose(x, z, self.env.agent.dir)
                self.current_idx = i
                return

    def _choose_start_edge_index(self) -> int:
        ax, az = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        dists = []
        for i, (x, z) in enumerate(self.edge_points):
            d = (x - ax) ** 2 + (z - az) ** 2
            dists.append((d, i))
        dists.sort()
        return dists[0][1]

    def _turn_toward(self, desired: float) -> Optional[int]:
        a = self.env.actions
        curr = float(self.env.agent.dir)
        err = abs(self._wrap(desired - curr))
        if err <= self.turn_step_rad * 0.5:
            return None
        left_err = abs(self._wrap(desired - (curr + self.turn_step_rad)))
        right_err = abs(self._wrap(desired - (curr - self.turn_step_rad)))
        return a.turn_left if left_err <= right_err else a.turn_right

    def _desired_inward_dir(self) -> float:
        x, z = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        return math.atan2(-(self.cz - z), (self.cx - x))

    def _next_idx(self) -> int:
        return (self.current_idx + 1) % 4

    def _choose_next_edge_random(self) -> int:
        candidates = [0, 1, 2, 3]
        try:
            candidates.remove(self.current_idx)
        except ValueError:
            pass
        j = int(self.rng.integers(0, len(candidates)))
        return int(candidates[j])

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent
        ax, az = float(agent.pos[0]), float(agent.pos[2])

        if self.phase == "align_inward":
            desired = self._desired_inward_dir()
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "observe_inward"
            self.observe_remaining = self.observe_inward_steps
            return a.do_nothing

        if self.phase == "observe_inward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.do_nothing
            self.phase = "align_outward"

        if self.phase == "align_outward":
            inward = self._desired_inward_dir()
            desired = self._wrap(inward + math.pi)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "observe_outward"
            self.observe_remaining = self.observe_outward_steps
            return a.do_nothing

        if self.phase == "observe_outward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.do_nothing
            self.phase = "align_to_center"

        if self.phase == "align_to_center":
            desired = self._dir_to(self.cx, self.cz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_center"
            return a.do_nothing

        if self.phase == "forward_to_center":
            dx, dz = (self.cx - ax), (self.cz - az)
            dist = math.hypot(dx, dz)
            if dist <= self.reach_eps:
                self._set_agent_pose(self.cx, self.cz, agent.dir)
                self.phase = "align_to_edge"
                return a.do_nothing
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.do_nothing

        if self.phase == "align_to_edge":
            if self.target_idx is None:
                self.target_idx = self._choose_next_edge_random()
            tx, tz = self.edge_points[self.target_idx]
            desired = self._dir_to(tx, tz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_edge"
            return a.do_nothing

        if self.phase == "forward_to_edge":
            tx, tz = self.edge_points[self.target_idx]
            dx, dz = (tx - ax), (tz - az)
            dist = math.hypot(dx, dz)
            if dist <= self.reach_eps:
                self._set_agent_pose(tx, tz, agent.dir)
                self.current_idx = self.target_idx
                self.target_idx = None
                self.phase = "align_inward"
                self.observe_remaining = self.observe_inward_steps
                return a.do_nothing
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.do_nothing

        return a.do_nothing

class PeeakbooPolicy:
    """
    Stationary: spawn at the midpoint of a random wall, then alternate:
      - align inward to room, NOOP for observe_steps
      - align outward toward the wall, NOOP for observe_steps
    Repeats indefinitely; emits only NOOP except for discrete turns to align.
    """

    def __init__(self, env: gym.Env, observe_steps: int = 70):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.observe_steps = int(max(0, observe_steps))

        # Room geometry
        self.cx = float((self.env.min_x + self.env.max_x) * 0.5)
        self.cz = float((self.env.min_z + self.env.max_z) * 0.5)
        wall_buf = float(getattr(self.env, "spawn_wall_buffer", 1.0))

        # Wall midpoints (N, E, S, W)
        self.wall_points = [
            (self.cx, self.env.min_z + wall_buf),
            (self.env.max_x - wall_buf, self.cz),
            (self.cx, self.env.max_z - wall_buf),
            (self.env.min_x + wall_buf, self.cz),
        ]

        # Controller params
        turn_step_deg = float(self.env.params.get_max("turn_step"))
        self.turn_step_rad = turn_step_deg * math.pi / 180.0

        # Choose a wall midpoint that is free, preferring nearest to current
        self._try_snap_to_random_wall()
        # Start aligned inward
        self.phase = "align_inward"
        self.observe_remaining = self.observe_steps

    def _wrap(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _dir_to(self, x: float, z: float) -> float:
        ax = float(self.env.agent.pos[0])
        az = float(self.env.agent.pos[2])
        dx = x - ax
        dz = z - az
        return math.atan2(-dz, dx)

    def _is_pos_free(self, x: float, z: float) -> bool:
        agent = self.env.agent
        pos = agent.pos.copy()
        pos[0] = float(x)
        pos[2] = float(z)
        return not bool(self.env.intersect(agent, pos, agent.radius))

    def _set_agent_pose(self, x: float, z: float, dir_rad: float):
        self.env.agent.pos[0] = float(x)
        self.env.agent.pos[2] = float(z)
        self.env.agent.dir = float(self._wrap(dir_rad))

    def _try_snap_to_random_wall(self):
        # order by distance to current, then pick the first free; if none free, keep current
        ax, az = float(self.env.agent.pos[0]), float(self.env.agent.pos[2])
        candidates = []
        for i, (x, z) in enumerate(self.wall_points):
            d = (x - ax) ** 2 + (z - az) ** 2
            candidates.append((d, i, x, z))
        candidates.sort()
        # optionally shuffle among equal distances using RNG
        for _, _, x, z in candidates:
            if self._is_pos_free(x, z):
                # keep current heading; alignment handled by phases
                self._set_agent_pose(x, z, self.env.agent.dir)
                return
        # otherwise, do nothing

    def _turn_toward(self, desired: float) -> Optional[int]:
        a = self.env.actions
        curr = float(self.env.agent.dir)
        err = abs(self._wrap(desired - curr))
        # Accept alignment if within half a turn step plus a tiny epsilon to avoid oscillations
        if err <= (self.turn_step_rad * 0.5 + 1e-3):
            return None
        left_err = abs(self._wrap(desired - (curr + self.turn_step_rad)))
        right_err = abs(self._wrap(desired - (curr - self.turn_step_rad)))
        return a.turn_left if left_err <= right_err else a.turn_right

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        # inward is toward center; outward is 180 degrees opposite
        inward_dir = self._dir_to(self.cx, self.cz)
        outward_dir = self._wrap(inward_dir + math.pi)

        if self.phase == "align_inward":
            turn = self._turn_toward(inward_dir)
            if turn is not None:
                return turn
            self.phase = "observe_inward"
            self.observe_remaining = self.observe_steps
            return a.do_nothing

        if self.phase == "observe_inward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.do_nothing
            self.phase = "align_outward"
            return a.do_nothing

        if self.phase == "align_outward":
            turn = self._turn_toward(outward_dir)
            if turn is not None:
                return turn
            self.phase = "observe_outward"
            self.observe_remaining = self.observe_steps
            return a.do_nothing

        if self.phase == "observe_outward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.do_nothing
            self.phase = "align_inward"
            return a.do_nothing

        return a.do_nothing

class BiasedWalkV2Policy:
    """
    Three-phase wall-biased exploration that cycles:
      0) On spawn, walk straight until hitting a wall
      1) "look": turn to face the room center and pause for observe_steps (NOOP)
      2) "wall crawl": pick left or right along the wall, move forward with probability forward_prob; when we decide to turn, align inward toward the room and transition to (3)
      3) "walk in room": move forward with probability forward_prob; when we decide to turn, make a discrete turn and then walk straight until we hit the wall; on wall impact, turn around and return to (1)

    Notes:
      - Uses env.max_forward_step and env.params.turn_step for discrete motions
      - NOOP is implemented as env.pickup action id
    """

    def __init__(self, env: gym.Env, forward_prob: float = 0.8, observe_steps: int = 5, debug: bool = False, debug_writer=None):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.forward_prob = float(forward_prob)
        self.observe_steps = int(max(0, observe_steps))
        self.debug = bool(debug)
        self.debug_writer = debug_writer

        # Room center
        self.cx = float((self.env.min_x + self.env.max_x) * 0.5)
        self.cz = float((self.env.min_z + self.env.max_z) * 0.5)

        # Discrete turn step in radians
        turn_step_deg = float(self.env.params.get_max("turn_step"))
        self.turn_step_rad = turn_step_deg * math.pi / 180.0
        self.align_eps_rad = math.radians(5.0)

        # State machine
        self.phase = "spawn_to_wall"
        self.look_remaining = self.observe_steps
        self.crawl_sign = 0  # +1 for left, -1 for right
        self.target_dir = None

    def _dist_to_walls(self, pos: np.ndarray) -> float:
        x, _, z = pos
        return min(
            x - self.env.min_x,
            self.env.max_x - x,
            z - self.env.min_z,
            self.env.max_z - z,
        )

    def _wrap(self, a: float) -> float:
        return (a + math.pi) % (2 * math.pi) - math.pi

    def _dir_to(self, x: float, z: float) -> float:
        ax = float(self.env.agent.pos[0])
        az = float(self.env.agent.pos[2])
        dx = x - ax
        dz = z - az
        return math.atan2(-dz, dx)

    def _ahead_pos(self, pos: np.ndarray, dir_rad: float, dist: float) -> np.ndarray:
        dx = math.cos(dir_rad) * dist
        dz = -math.sin(dir_rad) * dist
        nxt = pos.copy()
        nxt[0] += dx
        nxt[2] += dz
        return nxt

    def _forward_blocked(self) -> bool:
        agent = self.env.agent
        fwd_step = float(self.env.max_forward_step)
        # Consider possible lateral drifts that will be sampled at step-time.
        # We don't sample RNG here (to avoid advancing it); instead, we conservatively
        # require forward to be free for a small set of representative drift values.
        try:
            max_drift = float(abs(self.env.params.get_max("forward_drift")))
        except Exception:
            max_drift = 0.0
        # Representative drift candidates: center and extremes
        drift_candidates = [0.0]
        if max_drift > 0.0:
            drift_candidates.extend([-max_drift, max_drift])
        # Precompute heading vectors
        dir_rad = float(agent.dir)
        dx = math.cos(dir_rad)
        dz = -math.sin(dir_rad)
        # Right vector in XZ plane (rotate +90deg from forward)
        rx = -dz
        rz = dx
        for drift in drift_candidates:
            next_pos = agent.pos.copy()
            next_pos[0] += dx * fwd_step + rx * drift
            next_pos[2] += dz * fwd_step + rz * drift
            hit = self.env.intersect(agent, next_pos, agent.radius)
            if hit:
                return True
            # If carrying an entity, ensure carry position would also be collision-free
            carrying = getattr(agent, "carrying", None)
            if carrying is not None:
                try:
                    next_carry = self.env._get_carry_pos(next_pos, carrying)
                    if self.env.intersect(carrying, next_carry, carrying.radius):
                        return True
                except Exception:
                    # If for any reason we can't compute carry pose, be conservative
                    return True
        return False

    def _log(self, msg: str):
        if not self.debug:
            return
        if self.debug_writer is not None:
            self.debug_writer(msg)
        else:
            print(msg)

    def _turn_toward(self, desired: float) -> Optional[int]:
        a = self.env.actions
        curr = float(self.env.agent.dir)
        err = abs(self._wrap(desired - curr))
        if err <= (self.turn_step_rad * 0.5 + self.align_eps_rad):
            return None
        # choose left/right turn that reduces error
        left_err = abs(self._wrap(desired - (curr + self.turn_step_rad)))
        right_err = abs(self._wrap(desired - (curr - self.turn_step_rad)))
        return a.turn_left if left_err <= right_err else a.turn_right

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent
        # Optional debug snapshot
        fwd_blocked = self._forward_blocked()
        center_dir = self._dir_to(self.cx, self.cz)
        self._log(
            f"[BWv2] t={step_idx:04d} phase={self.phase} pos=({float(agent.pos[0]):.2f},{float(agent.pos[2]):.2f}) "
            f"dir={math.degrees(float(agent.dir))%360:.1f} deg center_dir={math.degrees(center_dir)%360:.1f} "
            f"fwd_blocked={int(fwd_blocked)} crawl_sign={self.crawl_sign}"
        )

        # 0) Spawn: go straight until we hit a wall
        if self.phase == "spawn_to_wall":
            if not self._forward_blocked():
                return a.move_forward
            self.phase = "look_align"
            self.look_remaining = self.observe_steps
            return a.do_nothing

        # 1) Look: face the center and pause
        if self.phase == "look_align":
            desired = self._dir_to(self.cx, self.cz)
            turn = self._turn_toward(desired)
            if turn is not None:
                self._log(f"[BWv2] action=turn ({'L' if turn==a.turn_left else 'R'}) in look_align")
                return turn
            self.phase = "look_observe"
            self.look_remaining = self.observe_steps
            self._log("[BWv2] action=NOOP enter look_observe")
            return a.do_nothing

        if self.phase == "look_observe":
            if self.look_remaining > 0:
                self.look_remaining -= 1
                return a.do_nothing
            # choose crawl side and align along wall (left/right relative to inward)
            self.crawl_sign = +1 if float(self.rng.random()) < 0.5 else -1
            self.phase = "wall_crawl_align"
            self._log("[BWv2] action=NOOP enter wall_crawl_align")
            return a.do_nothing

        # 2) Wall crawl
        if self.phase == "wall_crawl_align":
            center_dir = self._dir_to(self.cx, self.cz)
            desired = self._wrap(center_dir + (self.crawl_sign * (math.pi / 2.0)))
            # If we can step forward, treat as aligned (prevents corner oscillation)
            if not self._forward_blocked():
                self.phase = "wall_crawl_move"
                self._log("[BWv2] forward free in wall_crawl_align -> enter wall_crawl_move")
                return a.do_nothing
            turn = self._turn_toward(desired)
            if turn is not None:
                self._log(f"[BWv2] action=turn ({'L' if turn==a.turn_left else 'R'}) in wall_crawl_align")
                return turn
            self.phase = "wall_crawl_move"
            self._log("[BWv2] action=NOOP enter wall_crawl_move")
            return a.do_nothing

        if self.phase == "wall_crawl_move":
            # With forward_prob move along the wall; if blocked, turn to keep hugging
            if float(self.rng.random()) < self.forward_prob and not self._forward_blocked():
                self._log("[BWv2] action=forward along wall")
                return a.move_forward
            if self._forward_blocked():
                # left-hand rule: turn right; right-hand rule: turn left
                # Log predicted blockage after a single turn in each direction
                turn_step_rad = self.turn_step_rad
                curr_dir = float(agent.dir)
                # predict next forward blocked if we turn left/right once
                left_dir = curr_dir + turn_step_rad
                right_dir = curr_dir - turn_step_rad
                fwd_step = float(self.env.max_forward_step)
                left_next = self._ahead_pos(agent.pos, left_dir, fwd_step)
                right_next = self._ahead_pos(agent.pos, right_dir, fwd_step)
                left_blocked = bool(self.env.intersect(agent, left_next, agent.radius))
                right_blocked = bool(self.env.intersect(agent, right_next, agent.radius))
                self._log(f"[BWv2] corner: fwd_blocked=1 left_next_blocked={int(left_blocked)} right_next_blocked={int(right_blocked)}")
                act = a.turn_right if self.crawl_sign > 0 else a.turn_left
                self._log(f"[BWv2] action=turn ({'R' if act==a.turn_right else 'L'}) in corner")
                return act
            # decide to turn inward
            self.target_dir = self._dir_to(self.cx, self.cz)
            self.phase = "walk_room_align"
            self._log("[BWv2] action=NOOP enter walk_room_align")
            return a.do_nothing

        # 3) Walk in room
        if self.phase == "walk_room_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                self._log(f"[BWv2] action=turn ({'L' if turn==a.turn_left else 'R'}) in walk_room_align")
                return turn
            self.phase = "walk_room_move"
            self._log("[BWv2] action=NOOP enter walk_room_move")
            return a.do_nothing

        if self.phase == "walk_room_move":
            if float(self.rng.random()) < self.forward_prob and not self._forward_blocked():
                self._log("[BWv2] action=forward in room")
                return a.move_forward
            # commit to a discrete turn and then go straight to the wall
            turn_sign = +1 if float(self.rng.random()) < 0.5 else -1
            curr_dir = float(agent.dir)
            self.target_dir = self._wrap(curr_dir + turn_sign * self.turn_step_rad)
            self.phase = "go_to_wall_align"
            self._log("[BWv2] action=NOOP enter go_to_wall_align")
            return a.do_nothing

        if self.phase == "go_to_wall_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                self._log(f"[BWv2] action=turn ({'L' if turn==a.turn_left else 'R'}) in go_to_wall_align")
                return turn
            self.phase = "go_to_wall_move"
            self._log("[BWv2] action=NOOP enter go_to_wall_move")
            return a.do_nothing

        if self.phase == "go_to_wall_move":
            if not self._forward_blocked():
                self._log("[BWv2] action=forward to wall")
                return a.move_forward
            # Hit wall: turn around, then go to look phase
            self.target_dir = self._wrap(float(agent.dir) + math.pi)
            self.phase = "turn_around_align"
            self._log("[BWv2] action=NOOP enter turn_around_align")
            return a.do_nothing

        if self.phase == "turn_around_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                self._log(f"[BWv2] action=turn ({'L' if turn==a.turn_left else 'R'}) in turn_around_align")
                return turn
            self.phase = "look_align"
            self.look_remaining = self.observe_steps
            self._log("[BWv2] action=NOOP enter look_align")
            return a.do_nothing

        # fallback
        return a.do_nothing

def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def _wrap_angle_0_2pi(a: float) -> float:
    """Wrap an angle in radians to [0, 2π)."""
    two_pi = 2.0 * math.pi
    return float(a % two_pi)


def _agent_frame_components(agent_dir: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 2D forward/right unit vectors in the XZ plane for each agent_dir (radians).

    MiniWorld convention:
      dir_vec = (cos(dir), 0, -sin(dir))  => forward2 = (cos, -sin) in (x,z)
      right_vec = (sin(dir), 0, cos(dir)) => right2  = (sin, cos)  in (x,z)
    """
    c = np.cos(agent_dir)
    s = np.sin(agent_dir)
    forward2 = np.stack([c, -s], axis=-1)  # (...,2)
    right2 = np.stack([s, c], axis=-1)     # (...,2)
    return forward2, right2


def _to_agent_frame(delta_xz: np.ndarray, agent_dir: np.ndarray) -> np.ndarray:
    """
    Convert world XZ deltas into the agent-normalized frame where the agent is at (0,0)
    and facing +X ("east").

    Returns rel_xz (...,2) where:
      rel_x = "in front of the agent"
      rel_z = "to the agent's right"
    """
    forward2, right2 = _agent_frame_components(agent_dir)
    rel_x = np.sum(delta_xz * forward2, axis=-1, keepdims=True)
    rel_z = np.sum(delta_xz * right2, axis=-1, keepdims=True)
    return np.concatenate([rel_x, rel_z], axis=-1)


def run_rollout(
    env: gym.Env,
    steps: int,
    align_heading_zero: bool,
    segment_len: int,
    policy_name: str,
    policy_kwargs: dict,
    observe_steps: int = 5,
    capture_top: bool = False,
    store_block_info: bool = False,
) -> Tuple[
    np.ndarray,  # rgb
    np.ndarray,  # depth
    np.ndarray,  # actions
    np.ndarray,  # top or None
    np.ndarray,  # agent_pos (T+1,3)
    np.ndarray,  # delta_xz (T,2)
    np.ndarray,  # delta_dir (T,)
    np.ndarray,  # agent_dir (T,)
    Optional[dict],  # top_view_scale (or None)
    Optional[dict],  # block_info (or None)
]:
    obs_list = []
    depth_list = []
    top_list = [] if capture_top else None
    top_view_scale: Optional[dict] = None
    actions = []
    agent_pos_list = []
    agent_dir_list = []
    # Block info capture (full trajectory length T+1, trimmed to T on return)
    block_pos_list = []

    obs, _ = env.reset()
    if align_heading_zero:
        env.unwrapped.agent.dir = 0.0
    # Always keep heading in [0, 2π)
    env.unwrapped.agent.dir = _wrap_angle_0_2pi(env.unwrapped.agent.dir)

    # Instantiate policy BEFORE first render so it can adjust spawn pose
    if policy_name == "back_and_forth":
        policy = BackAndForthPolicy(segment_len=segment_len)
    elif policy_name == "center_rotate":
        policy = CenterRotatePolicy(env=env)
    elif policy_name == "do_nothing":
        policy = DoNothingPolicy(env=env)
    elif policy_name == "edge_plus":
        policy = EdgePlusPolicy(env=env, observe_steps=observe_steps)
    elif policy_name == "biased_walk_v2":
        policy = BiasedWalkV2Policy(
            env=env,
            forward_prob=policy_kwargs.get("forward_prob", 0.8),
            observe_steps=observe_steps,
            debug=policy_kwargs.get("debug", False),
        )
    elif policy_name == "peekaboo_motion":
        policy = PeekabooMotionPolicy(
            env=env,
            observe_inward_steps=policy_kwargs.get("observe_inward_steps", observe_steps),
            observe_outward_steps=policy_kwargs.get("observe_outward_steps", max(1, observe_steps * 4)),
        )
    elif policy_name == "peeakboo":
        policy = PeeakbooPolicy(env=env, observe_steps=observe_steps)
    elif policy_name == "blockmover":
        policy = BlockMoverPolicy(env=env)
    else:
        policy = BiasedRandomPolicy(env=env, **policy_kwargs)

    # Policy may have adjusted spawn pose; ensure heading in [0, 2π) before first capture
    env.unwrapped.agent.dir = _wrap_angle_0_2pi(env.unwrapped.agent.dir)
    rgb = env.render()  # render current obs after any policy-driven pose adjust

    # Collect first frame
    obs_list.append(rgb)
    depth = env.unwrapped.render_depth(env.unwrapped.vis_fb)
    depth_list.append(depth)
    # Initial agent state
    agent_pos_list.append(env.unwrapped.agent.pos.copy())
    agent_dir_list.append(_wrap_angle_0_2pi(env.unwrapped.agent.dir))

    # Blocks: define a stable ordering within the episode
    block_info_static = None
    blocks = None
    if store_block_info:
        try:
            blocks = [e for e in env.unwrapped.entities if isinstance(e, Box)]
        except Exception:
            blocks = []
        block_info_static = {
            "block_ids": list(range(len(blocks))),
            "block_colors": [getattr(b, "color", None) for b in blocks],
            "block_textures": [getattr(b, "tex_name", None) for b in blocks],
        }
        if len(blocks) == 0:
            block_pos_list.append(np.zeros((0, 3), dtype=np.float32))
        else:
            block_pos_list.append(np.stack([b.pos.copy() for b in blocks], axis=0).astype(np.float32))

    if capture_top:
        # Capture scale once on the first call
        top, scale = env.unwrapped.render_top_view(
            env.unwrapped.vis_fb, render_agent=True, return_scale=True
        )
        top_list.append(top)
        top_view_scale = {
            "x_scale": float(scale["x_scale"]),
            "z_scale": float(scale["z_scale"]),
            "x_offset": float(scale["x_offset"]),
            "z_offset": float(scale["z_offset"]),
        }
    act_fn = policy.action
    for t in range(steps):
        a = act_fn(t)
        obs, reward, term, trunc, info = env.step(a)
        # MiniWorld accumulates heading; normalize to [0, 2π) for consistency
        env.unwrapped.agent.dir = _wrap_angle_0_2pi(env.unwrapped.agent.dir)
        rgb = env.render()
        obs_list.append(rgb)
        depth = env.unwrapped.render_depth(env.unwrapped.vis_fb)
        depth_list.append(depth)
        if capture_top:
            top = env.unwrapped.render_top_view(env.unwrapped.vis_fb, render_agent=True)
            top_list.append(top)
        actions.append(a)
        # Record agent state after step
        agent_pos_list.append(env.unwrapped.agent.pos.copy())
        agent_dir_list.append(_wrap_angle_0_2pi(env.unwrapped.agent.dir))
        if store_block_info:
            if blocks is None:
                blocks = [e for e in env.unwrapped.entities if isinstance(e, Box)]
            if len(blocks) == 0:
                block_pos_list.append(np.zeros((0, 3), dtype=np.float32))
            else:
                block_pos_list.append(np.stack([b.pos.copy() for b in blocks], axis=0).astype(np.float32))

        if term or trunc:
            break

    # Keep exactly one frame per executed transition (drop the extra terminal frame)
    steps_executed = len(actions)
    rgb_arr = np.stack(obs_list[:steps_executed], axis=0)  # (T,H,W,3)
    depth_arr = np.stack(depth_list[:steps_executed], axis=0)  # (T,H,W,1)
    actions_arr = np.array(actions, dtype=np.int64)  # (T,)
    top_arr = (
        np.stack(top_list[:steps_executed], axis=0) if (capture_top and top_list) else None
    )

    # Agent trajectories
    # Compute deltas from full trajectory (length T+1)
    agent_pos_full = np.stack(agent_pos_list, axis=0)
    agent_dir_full = np.array(agent_dir_list, dtype=np.float32)
    delta_xz = agent_pos_full[1: steps_executed + 1, (0, 2)] - agent_pos_full[
        :steps_executed, (0, 2)
    ]  # (T,2)
    delta_dir = _wrap_angle(
        agent_dir_full[1: steps_executed + 1] - agent_dir_full[:steps_executed]
    )  # (T,)
    # Trim absolute positions to match kept frames
    agent_pos = agent_pos_full[:steps_executed]  # (T,3)

    # Absolute heading per kept frame (length T)
    agent_dir = agent_dir_full[:steps_executed]

    block_info = None
    if store_block_info:
        # Full length is T+1; we convert into per-kept-frame (T) values
        if len(block_pos_list) != (steps_executed + 1):
            # Best-effort: align by truncation/padding
            block_pos_list = block_pos_list[: steps_executed + 1]
            while len(block_pos_list) < (steps_executed + 1):
                block_pos_list.append(block_pos_list[-1].copy() if block_pos_list else np.zeros((0, 3), dtype=np.float32))

        block_pos_full = np.stack(block_pos_list, axis=0).astype(np.float32)  # (T+1,N,3)
        block_pos = block_pos_full[:steps_executed]  # (T,N,3) aligned to kept frames
        block_vel_world = (block_pos_full[1: steps_executed + 1] - block_pos_full[:steps_executed]).astype(np.float32)  # (T,N,3)

        # Agent vel in world coords aligned to kept frames (T,3)
        agent_vel_world = (agent_pos_full[1: steps_executed + 1] - agent_pos_full[:steps_executed]).astype(np.float32)

        # Relative positions: subtract agent position and rotate into agent frame (XZ only)
        block_delta_xz = block_pos[:, :, (0, 2)] - agent_pos[:, (0, 2)][:, None, :]
        block_rel_xz = _to_agent_frame(block_delta_xz, agent_dir[:, None])  # (T,N,2)

        # Relative velocities: subtract agent velocity, then rotate into agent frame (XZ only)
        rel_vel_xz_world = block_vel_world[:, :, (0, 2)] - agent_vel_world[:, (0, 2)][:, None, :]
        block_rel_vel_xz = _to_agent_frame(rel_vel_xz_world, agent_dir[:, None])  # (T,N,2)

        block_info = {
            **(block_info_static or {}),
            "block_pos_world": torch.from_numpy(block_pos),             # (T,N,3)
            "block_vel_world": torch.from_numpy(block_vel_world),       # (T,N,3)
            "block_pos_agent": torch.from_numpy(block_rel_xz.astype(np.float32)),      # (T,N,2)
            "block_vel_agent": torch.from_numpy(block_rel_vel_xz.astype(np.float32)),  # (T,N,2)
            "agent_pos_world": torch.from_numpy(agent_pos.astype(np.float32)),         # (T,3)
            "agent_dir": torch.from_numpy(agent_dir.astype(np.float32)),               # (T,)
        }

    return rgb_arr, depth_arr, actions_arr, top_arr, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale, block_info


def write_mp4_rgb(out_path: str, frames: np.ndarray, fps: int = 15):
    import imageio

    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    # Use the FFmpeg writer; "quality" is not a valid ffmpeg arg. Use bitrate/CRF instead.
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        bitrate="8M",
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _generate_one(idx: int, ns: SimpleNamespace):
    # Construct per-item args namespace for env
    args = ns.args
    # Derive block directory and filenames
    block_id = idx // ns.block_size
    item_id = idx % ns.block_size
    block_dir = Path(ns.dataset_root) / f"{block_id}"
    block_dir.mkdir(parents=True, exist_ok=True)
    stem = str(item_id).zfill(ns.file_digits)
    out_prefix = str(block_dir / stem)

    # Seed per item for reproducibility
    # Choose a per-item seed
    if args.seed is None:
        # High-entropy seed using OS randomness
        import secrets

        item_seed = secrets.randbits(64)
    else:
        item_seed = int(args.seed) + idx

    # Build env with same knobs
    env_args = SimpleNamespace(**vars(args))
    env_args.out_prefix = out_prefix
    env_args.no_time_limit = True  # in dataset mode, default to ignore time limit

    # Make sure domain randomization can vary per item while still reproducible
    env = build_env(env_args)

    # Apply seed via gym API
    env.reset(seed=item_seed)

    policy_kwargs = dict(
        forward_prob=args.forward_prob,
        turn_left_weight=args.turn_left_weight,
        turn_right_weight=args.turn_right_weight,
        wall_buffer=args.wall_buffer,
        avoid_turning_into_walls=args.avoid_turning_into_walls,
        lookahead_mult=args.lookahead_mult,
        debug=args.debug_join,
        observe_inward_steps=(args.observe_inward_steps if args.observe_inward_steps is not None else args.observe_steps),
        observe_outward_steps=(args.observe_outward_steps if args.observe_outward_steps is not None else (4 * args.observe_steps)),
    )

    rgb, depth, actions, top, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale, block_info = run_rollout(
        env,
        args.steps,
        align_heading_zero=args.heading_zero,
        segment_len=args.segment_len,
        policy_name=args.policy,
        policy_kwargs=policy_kwargs,
        observe_steps=args.observe_steps,
        capture_top=(args.debug_join or args.output_2d_map),
        store_block_info=getattr(args, "store_block_info", False),
    )

    write_mp4_rgb(f"{out_prefix}_rgb.mp4", rgb)
    # Save raw depth (float32) without quantization
    torch.save(torch.from_numpy(depth).to(torch.float32), f"{out_prefix}_depth.pt")
    meta = {
        "actions": torch.tensor(actions, dtype=torch.long),
        "agent_pos": torch.tensor(agent_pos, dtype=torch.float32),
        "delta_xz": torch.tensor(delta_xz, dtype=torch.float32),
        "delta_dir": torch.tensor(delta_dir, dtype=torch.float32),
        "agent_dir": torch.tensor(agent_dir, dtype=torch.float32),
    }
    if args.output_2d_map and top_view_scale is not None:
        meta["top_view_scale"] = {k: float(v) for k, v in top_view_scale.items()}
    torch.save(meta, f"{out_prefix}_actions.pt")
    if args.debug_join and top is not None:
        H, W = rgb.shape[1], rgb.shape[2]
        if top.shape[1] != H or top.shape[2] != W:
            import cv2

            resized = [cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST) for f in top]
            top = np.stack(resized, axis=0)
        side_by_side = np.concatenate([rgb, top], axis=2)
        write_mp4_rgb(f"{out_prefix}_debug.mp4", side_by_side)

    if args.output_2d_map and top is not None:
        write_mp4_rgb(f"{out_prefix}_map_2d.mp4", top)

    if getattr(args, "store_block_info", False) and block_info is not None:
        torch.save(block_info, f"{out_prefix}_block_info.pt")

    env.close()
    return idx


def run_parallel_dataset(args):
    # DEPRECATED: in-process parallel dataset generation. Use scripts.generate_videos_batch instead.
    # Leaving this stub to avoid breaking callers; consider removing in future.
    raise RuntimeError(
        "Deprecated: Use 'python -m scripts.generate_videos_batch -- ...' to run multi-generation via subprocesses."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out-prefix", type=str, required=False)
    # Render (RGB) resolution
    parser.add_argument("--render-width", type=int, default=128)
    parser.add_argument("--render-height", type=int, default=128)
    # Observation/depth buffer resolution (used by depth unless overridden)
    parser.add_argument("--obs-width", type=int, default=128)
    parser.add_argument("--obs-height", type=int, default=128)
    parser.add_argument("--top_view", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--no-time-limit", action="store_true")
    parser.add_argument("--heading-zero", action="store_true")
    parser.add_argument("--segment-len", type=int, default=10, help="steps to move forward before turning around")

    # Movement/env customization flags (mirror manual_control)
    parser.add_argument("--turn-step-deg", type=float, default=None)
    parser.add_argument("--forward-step", type=float, default=None)
    parser.add_argument("--box-speed-scale", type=float, default=None)
    parser.add_argument("--box-allow-overlap", action="store_true")
    parser.add_argument("--agent-box-allow-overlap", action="store_true")
    parser.add_argument("--box-random-orientation", action="store_true")
    parser.add_argument("--blocks-static", action="store_true", help="do not move boxes; keep them static")
    parser.add_argument("--block-torus-wrap", action="store_true", help="blocks wrap across walls (torus); agent unchanged")
    parser.add_argument("--grid-mode", action="store_true")
    parser.add_argument("--grid-vel-min", type=int, default=-1)
    parser.add_argument("--grid-vel-max", type=int, default=1)
    parser.add_argument("--grid-cardinal-only", action="store_true", help="grid mode: restrict block motion to cardinal directions only")
    # Block count and color palette controls (MovingBlockWorld)
    parser.add_argument("--num-blocks", type=int, default=None, help="override number of blocks")
    parser.add_argument("--num-blocks-min", type=int, default=None, help="min number of blocks (sampled uniformly if max is set)")
    parser.add_argument("--num-blocks-max", type=int, default=None, help="max number of blocks (sampled uniformly if min is set)")
    parser.add_argument("--ensure-base-palette", action="store_true", help="ensure at least one of each base colors: green, red, yellow, blue, purple, gray; remainder random")
    parser.add_argument("--debug-join", dest="debug_join", action="store_true", help="save a side-by-side debug video with RGB (left) and top-view map (right)")
    parser.add_argument("--output-2d-map", dest="output_2d_map", action="store_true", help="save the top-view map as a separate MP4 named *_map_2d.mp4")
    parser.add_argument(
        "--store-block-info",
        dest="store_block_info",
        action="store_true",
        help="save per-timestep block metadata to <out-prefix>_block_info.pt (positions/velocities in world + agent-normalized frame, plus agent pose)",
    )
    parser.add_argument("--spawn-wall-buffer", type=float, default=1.0, help="extra buffer from walls when spawning agent and boxes (meters)")
    parser.add_argument("--room-size", type=int, default=12, help="square room side length in meters (e.g., 12)")
    parser.add_argument("--cam-fov-y", type=float, default=None, help="camera vertical field of view in degrees (locks to this value if provided)")
    parser.add_argument("--even-lighting", action="store_true", help="use uniform ambient lighting (no directional shading)")
    parser.add_argument("--floor-tex", type=str, default="concrete", help="floor texture name (see miniworld/textures), default white")
    # parser.add_argument("--floor-tex", type=str, default="ceiling_tile_noborder", help="floor texture name (see miniworld/textures), default white")
    parser.add_argument("--wall-tex", type=str, default="white", help="wall texture name, default white")
    parser.add_argument("--ceil-tex", type=str, default="ceiling_tile_noborder", help="ceiling texture name, default white")
    parser.add_argument("--box-tex", type=str, default=None, help="box texture name (e.g., 'airduct_grate'); default none for solid color")
    parser.add_argument("--randomize-box-tex", action="store_true", help="assign a random texture per box from a default pool")
    parser.add_argument("--randomize-wall-tex", action="store_true", help="randomize wall texture from default pool ['brick_wall','drywall','wood']")
    parser.add_argument("--randomize-floor-tex", action="store_true", help="randomize floor texture from default pool ['cardboard','grass','concrete']")
    parser.add_argument("--box-and-ball", action="store_true", help="randomly spawn boxes or tall textured balls (height follows block-height)")
    # parser.add_argument("--ceil-tex", type=str, default="wood", help="ceiling texture name, default white")
    # Block size controls (MovingBlockWorld): uniform footprint and/or height
    parser.add_argument("--block-size-xy", type=float, default=None, help="if set, use this x/z size (meters) for all blocks")
    parser.add_argument("--block-height", type=float, default=None, help="if set, use this height (meters) for all blocks")

    # Parallel dataset generation (DEPRECATED): use scripts.generate_videos_batch
    parser.add_argument("--dataset-root", type=str, default=None, help="DEPRECATED: use scripts.generate_videos_batch for multi-generation")
    parser.add_argument("--num-videos", type=int, default=0, help="DEPRECATED")
    parser.add_argument("--block-size", type=int, default=100, help="DEPRECATED")
    parser.add_argument("--num-processes", type=int, default=4, help="DEPRECATED")
    parser.add_argument("--file-digits", type=int, default=4, help="DEPRECATED")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed for deterministic generation across items; if omitted, each item uses a random seed",
    )

    # Policy selection and knobs
    parser.add_argument("--policy", type=str, default="biased_random", choices=["biased_random", "biased_walk_v2", "back_and_forth", "center_rotate", "do_nothing", "edge_plus", "peeakboo", "peekaboo_motion", "blockmover"], help="which control policy to use")
    parser.add_argument("--forward-prob", type=float, default=0.8, help="biased_random: probability of moving forward when safe")
    parser.add_argument("--turn-left-weight", type=float, default=1.0, help="biased_random: relative weight for choosing left turns")
    parser.add_argument("--turn-right-weight", type=float, default=1.0, help="biased_random: relative weight for choosing right turns")
    parser.add_argument("--wall-buffer", type=float, default=1.5, help="biased_random: minimum distance from walls before turning away")
    parser.add_argument("--avoid-turning-into-walls", action="store_true", help="biased_random: when turning, prefer direction that increases distance to walls")
    parser.add_argument("--lookahead-mult", type=float, default=2.0, help="biased_random: lookahead distance multiplier relative to max forward step")
    # Agent spawn option: place agent at the center (uses env support)
    parser.add_argument("--agent-center-start", action="store_true", help="spawn the agent at the room center (top-left of middle for even sizes)")
    # EdgePlus/Peekaboo observation duration
    parser.add_argument("--observe-steps", type=int, default=5, help="edge_plus: number of NOOP steps to observe at each edge center; also used as default inward observe for peekaboo_motion")
    parser.add_argument("--observe-inward-steps", type=int, default=None, help="peekaboo_motion: number of NOOP steps to observe inward at edge (defaults to --observe-steps)")
    parser.add_argument("--observe-outward-steps", type=int, default=None, help="peekaboo_motion: number of NOOP steps to observe outward at edge (defaults to 4x inward)")

    args = parser.parse_args()

    # If dataset_root is provided, run parallel generation
    if args.dataset_root:
        run_parallel_dataset(args)
        return

    env = build_env(args)
    print(f"Miniworld v{miniworld.__version__}, Env: {args.env_name}")

    policy_kwargs = dict(
        forward_prob=args.forward_prob,
        turn_left_weight=args.turn_left_weight,
        turn_right_weight=args.turn_right_weight,
        wall_buffer=args.wall_buffer,
        avoid_turning_into_walls=args.avoid_turning_into_walls,
        lookahead_mult=args.lookahead_mult,
        debug=args.debug_join,
        observe_inward_steps=(args.observe_inward_steps if getattr(args, "observe_inward_steps", None) is not None else args.observe_steps),
        observe_outward_steps=(args.observe_outward_steps if getattr(args, "observe_outward_steps", None) is not None else (4 * args.observe_steps)),
    )

    rgb, depth, actions, top, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale, block_info = run_rollout(
        env,
        args.steps,
        align_heading_zero=args.heading_zero,
        segment_len=args.segment_len,
        policy_name=args.policy,
        policy_kwargs=policy_kwargs,
        observe_steps=args.observe_steps,
        capture_top=(args.debug_join or args.output_2d_map),
        store_block_info=getattr(args, "store_block_info", False),
    )

    # Save outputs
    write_mp4_rgb(f"{args.out_prefix}_rgb.mp4", rgb)
    # Save raw depth (float32) without quantization
    torch.save(torch.from_numpy(depth).to(torch.float32), f"{args.out_prefix}_depth.pt")
    meta = {
        "actions": torch.tensor(actions, dtype=torch.long),
        "agent_pos": torch.tensor(agent_pos, dtype=torch.float32),
        "delta_xz": torch.tensor(delta_xz, dtype=torch.float32),
        "delta_dir": torch.tensor(delta_dir, dtype=torch.float32),
        "agent_dir": torch.tensor(agent_dir, dtype=torch.float32),
    }
    if args.output_2d_map and top_view_scale is not None:
        # Save mapping to convert world (x,z) -> pixel (u,v)
        meta["top_view_scale"] = {
            k: float(v) for k, v in top_view_scale.items()
        }
    torch.save(meta, f"{args.out_prefix}_actions.pt")

    if args.debug_join and top is not None:
        # Concatenate RGB (left) and Top (right)
        # Ensure both are the same H,W
        H, W = rgb.shape[1], rgb.shape[2]
        if top.shape[1] != H or top.shape[2] != W:
            # Resize top to match RGB using simple nearest-neighbor
            import cv2

            resized = []
            for f in top:
                resized.append(cv2.resize(f, (W, H), interpolation=cv2.INTER_NEAREST))
            top = np.stack(resized, axis=0)

        side_by_side = np.concatenate([rgb, top], axis=2)
        write_mp4_rgb(f"{args.out_prefix}_debug.mp4", side_by_side)

    if args.output_2d_map and top is not None:
        write_mp4_rgb(f"{args.out_prefix}_map_2d.mp4", top)

    if getattr(args, "store_block_info", False) and block_info is not None:
        torch.save(block_info, f"{args.out_prefix}_block_info.pt")

    env.close()


if __name__ == "__main__":
    main()


