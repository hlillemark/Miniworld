#!/usr/bin/env python3

"""
Generate rollout videos (RGB and depth) and actions for a simple agent policy.

Example usage (mirrors manual_control flags):

python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --box-allow-overlap --agent-box-allow-overlap \
  --box-random-orientation --no-time-limit \
  --steps 300 --out-prefix ./out/run1

Outputs:
  - <out-prefix>_rgb.mp4
  - <out-prefix>_depth.mp4
  - <out-prefix>_actions.pt
  
  
command 3pm sep 10
single generation (static, center rotate):
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --forward-prob 0.90 --wall-buffer 0.0 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 300 --out-prefix ./out/run_move --debug-join --output-2d-map --room-size 7 \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy do_nothing --cam-fov-y 90


single generation (dynamic, edge plus agent): (edge plus)
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 300 --out-prefix ./out/edge_plus_run --debug-join --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy edge_plus --observe-steps 5 --cam-fov-y 60
  
  
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --out-prefix ./out/biased_random_fwd --debug-join --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_random --forward-prob 0.9 --cam-fov-y 60


single static generation
add --blocks-static
  
multi generation: 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 16 --no-time-limit \
  --dataset-root ./out/multi_gen --num-videos 20000 --block-size 256 --num-processes 64
  
# IMPORTANT NOTE FOR HEADLESS MODE RENDERING ON SERVERS: 
run like: 
MINIWORLD_HEADLESS=1 python -m scripts.generate_videos ... 

# NOTE: extra vae training samples generated like this: 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0 --avoid-turning-into-walls --agent-box-allow-overlap \
  --turn-step-deg 45 --forward-step 0.5 \
  --grid-vel-min -1 --grid-vel-max 1 --box-random-orientation \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 12 --no-time-limit \
  --dataset-root ./out/blockworld_futureproof --num-videos 20000 --block-size 256 --num-processes 64

import torch, torchvision.io as io; vid_depth = io.read_video("/Users/hansen/Desktop/ucsd/Miniworld/out/run_move_rgb.mp4", pts_unit="sec")[0].permute(0,3,1,2).to(torch.float32).div_(255)

# static generation for dfot map experiment 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 --blocks-static \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 16 --no-time-limit --output-2d-map \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/static_training_w_map --num-videos 20000 --block-size 256 --num-processes 32

# static generation center rotate with map
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 90 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/static_center_rotate_training --num-videos 20000 --block-size 256 --num-processes 32
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 90 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/static_center_rotate_validation --num-videos 1000 --block-size 64 --num-processes 32
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
        return a.pickup  # id 4 used as NOOP in datasets
    

class DoNothingPolicy:
    """
    action id 4 is used as noop
    Does nothing, just stands there.
    """

    def __init__(self, env: gym.Env):
        self.env = env.unwrapped

    def action(self, step_idx: int) -> int:
        a = self.env.actions
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
            return a.pickup

        if self.phase == "observe_edge":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.pickup
            # proceed to center route
            self.phase = "align_to_center"

        if self.phase == "align_to_center":
            desired = self._dir_to(self.cx, self.cz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_center"
            return a.pickup

        if self.phase == "forward_to_center":
            dx, dz = (self.cx - ax), (self.cz - az)
            dist = math.hypot(dx, dz)
            if dist <= self.reach_eps:
                # snap to exact center and proceed to align toward next edge
                self._set_agent_pose(self.cx, self.cz, agent.dir)
                self.phase = "align_to_edge"
                return a.pickup
            # drive straight; if blocked by dynamic obstacle, wait
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.pickup

        if self.phase == "align_to_edge":
            if self.target_idx is None:
                self.target_idx = self._choose_next_edge_random()
            tx, tz = self.edge_points[self.target_idx]
            desired = self._dir_to(tx, tz)
            turn = self._turn_toward(desired)
            if turn is not None:
                return turn
            self.phase = "forward_to_edge"
            return a.pickup

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
                return a.pickup
            fwd_step = float(self.env.max_forward_step)
            ahead_x = ax + math.cos(float(agent.dir)) * fwd_step
            ahead_z = az - math.sin(float(agent.dir)) * fwd_step
            if self._is_pos_free(ahead_x, ahead_z):
                return a.move_forward
            return a.pickup

        # fallback
        return a.pickup


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
        if err <= self.turn_step_rad * 0.5:
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
            return a.pickup

        if self.phase == "observe_inward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.pickup
            self.phase = "align_outward"
            return a.pickup

        if self.phase == "align_outward":
            turn = self._turn_toward(outward_dir)
            if turn is not None:
                return turn
            self.phase = "observe_outward"
            self.observe_remaining = self.observe_steps
            return a.pickup

        if self.phase == "observe_outward":
            if self.observe_remaining > 0:
                self.observe_remaining -= 1
                return a.pickup
            self.phase = "align_inward"
            return a.pickup

        return a.pickup

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

    def __init__(self, env: gym.Env, forward_prob: float = 0.8, observe_steps: int = 5):
        self.env = env.unwrapped
        self.rng = self.env.np_random
        self.forward_prob = float(forward_prob)
        self.observe_steps = int(max(0, observe_steps))

        # Room center
        self.cx = float((self.env.min_x + self.env.max_x) * 0.5)
        self.cz = float((self.env.min_z + self.env.max_z) * 0.5)

        # Discrete turn step in radians
        turn_step_deg = float(self.env.params.get_max("turn_step"))
        self.turn_step_rad = turn_step_deg * math.pi / 180.0

        # State machine
        self.phase = "spawn_to_wall"
        self.look_remaining = self.observe_steps
        self.crawl_sign = 0  # +1 for left, -1 for right
        self.target_dir = None
        # Stuck mitigation state
        self._last_pos = None
        self._prev_action = None
        self._turn_streak = 0
        self._prefer_forward_next = False

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
        next_pos = self._ahead_pos(agent.pos, float(agent.dir), fwd_step)
        return bool(self.env.intersect(agent, next_pos, agent.radius))

    def _turn_toward(self, desired: float) -> Optional[int]:
        a = self.env.actions
        curr = float(self.env.agent.dir)
        err = abs(self._wrap(desired - curr))
        if err <= self.turn_step_rad * 0.5:
            return None
        # choose left/right turn that reduces error
        left_err = abs(self._wrap(desired - (curr + self.turn_step_rad)))
        right_err = abs(self._wrap(desired - (curr - self.turn_step_rad)))
        return a.turn_left if left_err <= right_err else a.turn_right

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        agent = self.env.agent

        # Track movement and turning streak
        if self._last_pos is None:
            self._last_pos = agent.pos.copy()
        moved = float(np.linalg.norm(agent.pos[[(0, 2)]] - self._last_pos[[(0, 2)]])) > 1e-6
        if moved or self._prev_action == a.move_forward:
            self._turn_streak = 0
        elif self._prev_action in (a.turn_left, a.turn_right):
            self._turn_streak += 1

        def _emit(action_id: int) -> int:
            self._prev_action = action_id
            self._last_pos = agent.pos.copy()
            if action_id in (a.turn_left, a.turn_right):
                self._prefer_forward_next = True
            return action_id

        # 0) Spawn: go straight until we hit a wall
        if self.phase == "spawn_to_wall":
            if not self._forward_blocked():
                return _emit(a.move_forward)
            # transition to look
            self.phase = "look_align"
            self.look_remaining = self.observe_steps
            return _emit(a.pickup)

        # 1) Look: face the center and pause
        if self.phase == "look_align":
            desired = self._dir_to(self.cx, self.cz)
            turn = self._turn_toward(desired)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "look_observe"
            self.look_remaining = self.observe_steps
            return _emit(a.pickup)

        if self.phase == "look_observe":
            if self.look_remaining > 0:
                self.look_remaining -= 1
                return _emit(a.pickup)
            # choose crawl side and align along wall (left/right relative to inward)
            self.crawl_sign = +1 if float(self.rng.random()) < 0.5 else -1
            self.phase = "wall_crawl_align"
            return _emit(a.pickup)

        # 2) Wall crawl
        if self.phase == "wall_crawl_align":
            center_dir = self._dir_to(self.cx, self.cz)
            desired = self._wrap(center_dir + (self.crawl_sign * (math.pi / 2.0)))
            turn = self._turn_toward(desired)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "wall_crawl_move"
            return _emit(a.pickup)

        if self.phase == "wall_crawl_move":
            # Continue forward with forward_prob; otherwise start turning inward to walk into room
            if self._prefer_forward_next and not self._forward_blocked():
                self._prefer_forward_next = False
                return _emit(a.move_forward)
            if float(self.rng.random()) < self.forward_prob:
                if not self._forward_blocked():
                    return _emit(a.move_forward)
                # corner: turn to keep hugging wall (left-hand rule: turn right; right-hand: turn left)
                turn = a.turn_right if self.crawl_sign > 0 else a.turn_left
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            # decide to turn inward
            center_dir = self._dir_to(self.cx, self.cz)
            self.target_dir = center_dir
            self.phase = "walk_room_align"
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            # already aligned
            self.phase = "walk_room_move"
            return _emit(a.pickup)

        # 3) Walk in room
        if self.phase == "walk_room_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "walk_room_move"
            return _emit(a.pickup)

        if self.phase == "walk_room_move":
            if self._prefer_forward_next and not self._forward_blocked():
                self._prefer_forward_next = False
                return _emit(a.move_forward)
            if float(self.rng.random()) < self.forward_prob:
                if not self._forward_blocked():
                    return _emit(a.move_forward)
                # unexpected blockage inside room: fall back to go-to-wall behavior
            # commit to a discrete turn and then go straight to the wall
            turn_sign = +1 if float(self.rng.random()) < 0.5 else -1
            curr_dir = float(agent.dir)
            self.target_dir = self._wrap(curr_dir + turn_sign * self.turn_step_rad)
            self.phase = "go_to_wall_align"
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "go_to_wall_move"
            return _emit(a.pickup)

        if self.phase == "go_to_wall_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "go_to_wall_move"
            return _emit(a.pickup)

        if self.phase == "go_to_wall_move":
            if not self._forward_blocked():
                return _emit(a.move_forward)
            # Hit wall: turn around, then go to look phase
            self.target_dir = self._wrap(float(agent.dir) + math.pi)
            self.phase = "turn_around_align"
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            # already turned around
            self.phase = "look_align"
            self.look_remaining = self.observe_steps
            return _emit(a.pickup)

        if self.phase == "turn_around_align":
            turn = self._turn_toward(self.target_dir)
            if turn is not None:
                if self._turn_streak >= 6 and not self._forward_blocked():
                    return _emit(a.move_forward)
                return _emit(turn)
            self.phase = "look_align"
            self.look_remaining = self.observe_steps
            return _emit(a.pickup)

        # fallback: NOOP
        return _emit(a.pickup)

def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_rollout(
    env: gym.Env,
    steps: int,
    align_heading_zero: bool,
    segment_len: int,
    policy_name: str,
    policy_kwargs: dict,
    observe_steps: int = 5,
    capture_top: bool = False,
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
]:
    obs_list = []
    depth_list = []
    top_list = [] if capture_top else None
    top_view_scale: Optional[dict] = None
    actions = []
    agent_pos_list = []
    agent_dir_list = []

    obs, _ = env.reset()
    if align_heading_zero:
        env.unwrapped.agent.dir = 0.0

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
        policy = BiasedWalkV2Policy(env=env, forward_prob=policy_kwargs.get("forward_prob", 0.8), observe_steps=observe_steps)
    elif policy_name == "peeakboo":
        policy = PeeakbooPolicy(env=env, observe_steps=observe_steps)
    else:
        policy = BiasedRandomPolicy(env=env, **policy_kwargs)

    rgb = env.render()  # render current obs after any policy-driven pose adjust

    # Collect first frame
    obs_list.append(rgb)
    depth = env.unwrapped.render_depth(env.unwrapped.vis_fb)
    depth_list.append(depth)
    # Initial agent state
    agent_pos_list.append(env.unwrapped.agent.pos.copy())
    agent_dir_list.append(float(env.unwrapped.agent.dir))
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
        agent_dir_list.append(float(env.unwrapped.agent.dir))

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

    return rgb_arr, depth_arr, actions_arr, top_arr, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale


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
    )

    rgb, depth, actions, top, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale = run_rollout(
        env,
        args.steps,
        align_heading_zero=args.heading_zero,
        segment_len=args.segment_len,
        policy_name=args.policy,
        policy_kwargs=policy_kwargs,
        observe_steps=args.observe_steps,
        capture_top=(args.debug_join or args.output_2d_map),
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
    parser.add_argument("--debug-join", dest="debug_join", action="store_true", help="save a side-by-side debug video with RGB (left) and top-view map (right)")
    parser.add_argument("--output-2d-map", dest="output_2d_map", action="store_true", help="save the top-view map as a separate MP4 named *_map_2d.mp4")
    parser.add_argument("--spawn-wall-buffer", type=float, default=1.0, help="extra buffer from walls when spawning agent and boxes (meters)")
    parser.add_argument("--room-size", type=int, default=12, help="square room side length in meters (e.g., 12)")
    parser.add_argument("--cam-fov-y", type=float, default=None, help="camera vertical field of view in degrees (locks to this value if provided)")
    parser.add_argument("--even-lighting", action="store_true", help="use uniform ambient lighting (no directional shading)")
    parser.add_argument("--floor-tex", type=str, default="concrete", help="floor texture name (see miniworld/textures), default white")
    # parser.add_argument("--floor-tex", type=str, default="ceiling_tile_noborder", help="floor texture name (see miniworld/textures), default white")
    parser.add_argument("--wall-tex", type=str, default="white", help="wall texture name, default white")
    parser.add_argument("--ceil-tex", type=str, default="ceiling_tile_noborder", help="ceiling texture name, default white")
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
    parser.add_argument("--policy", type=str, default="biased_random", choices=["biased_random", "biased_walk_v2", "back_and_forth", "center_rotate", "do_nothing", "edge_plus", "peeakboo"], help="which control policy to use")
    parser.add_argument("--forward-prob", type=float, default=0.8, help="biased_random: probability of moving forward when safe")
    parser.add_argument("--turn-left-weight", type=float, default=1.0, help="biased_random: relative weight for choosing left turns")
    parser.add_argument("--turn-right-weight", type=float, default=1.0, help="biased_random: relative weight for choosing right turns")
    parser.add_argument("--wall-buffer", type=float, default=1.5, help="biased_random: minimum distance from walls before turning away")
    parser.add_argument("--avoid-turning-into-walls", action="store_true", help="biased_random: when turning, prefer direction that increases distance to walls")
    parser.add_argument("--lookahead-mult", type=float, default=2.0, help="biased_random: lookahead distance multiplier relative to max forward step")
    # Agent spawn option: place agent at the center (uses env support)
    parser.add_argument("--agent-center-start", action="store_true", help="spawn the agent at the room center (top-left of middle for even sizes)")
    # EdgePlus observation duration
    parser.add_argument("--observe-steps", type=int, default=5, help="edge_plus: number of NOOP steps to observe at each edge center")

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
    )

    rgb, depth, actions, top, agent_pos, delta_xz, delta_dir, agent_dir, top_view_scale = run_rollout(
        env,
        args.steps,
        align_heading_zero=args.heading_zero,
        segment_len=args.segment_len,
        policy_name=args.policy,
        policy_kwargs=policy_kwargs,
        observe_steps=args.observe_steps,
        capture_top=(args.debug_join or args.output_2d_map),
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

    env.close()


if __name__ == "__main__":
    main()


