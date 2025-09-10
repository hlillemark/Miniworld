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
  
  
command 7pm aug 20
single generation:
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --forward-prob 0.90 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 300 --out-prefix ./out/run_move --debug-join --output-2d-map --room-size 13 \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate 

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



static generation for dfot map experiment 

python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 --blocks-static \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 16 --no-time-limit --output-2d-map \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/static_training_w_map --num-videos 20000 --block-size 256 --num-processes 32

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
    if getattr(args, "blocks_static", False):
        env_kwargs["blocks_static"] = True
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
    # Even lighting: force ambient 1, diffuse 0 regardless of domain rand
    if getattr(args, "even_lighting", False):
        if params is None:
            params = DEFAULT_PARAMS.copy()
        params.set("light_ambient", [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        params.set("light_color", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    if params is not None:
        env_kwargs["params"] = params

    try:
        env = gym.make(args.env_name, **env_kwargs)
    except TypeError:
        # strip unsupported extras
        for k in [
            "box_speed_scale",
            "box_allow_overlap",
            "agent_box_allow_overlap",
            "box_random_orientation",
            "grid_mode",
            "grid_vel_min",
            "grid_vel_max",
            "floor_tex",
            "blocks_static",
            "wall_tex",
            "ceil_tex",
            "spawn_wall_buffer",
            "block_size_xy",
            "block_height",
            "agent_center_start",
        ]:
            env_kwargs.pop(k, None)
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
        curr_dist = self._dist_to_walls(agent.pos)

        # Forward candidate (one step)
        next_pos = self._ahead_pos(agent.pos, agent.dir, fwd_step)
        forward_collides = bool(self.env.intersect(agent, next_pos, agent.radius))
        need_turn = curr_dist < self.wall_buffer or forward_collides

        def turn_dir_score(turn_sign: int) -> float:
            # turn_sign: +1 left, -1 right
            turn_step_deg = self.env.params.get_max("turn_step")
            turn_step_rad = float(turn_step_deg) * math.pi / 180.0
            new_dir = agent.dir + (turn_step_rad if turn_sign > 0 else -turn_step_rad)
            ahead2 = self._ahead_pos(agent.pos, new_dir, lookahead)
            return self._dist_to_walls(ahead2)

        if need_turn:
            left_score = turn_dir_score(+1)
            right_score = turn_dir_score(-1)
            if left_score == right_score:
                # Tie-break using weights
                probs = np.array([self.turn_left_weight, self.turn_right_weight], dtype=float)
                probs = probs / probs.sum()
                return a.turn_left if self.rng.random() < probs[0] else a.turn_right
            return a.turn_left if left_score > right_score else a.turn_right

        # Otherwise, biased random: mostly forward if safe
        if (not forward_collides) and (self.rng.random() < self.forward_prob):
            return a.move_forward

        # Choose turn based on weights, but optionally avoid turning into walls
        if self.avoid_turning_into_walls:
            left_score = turn_dir_score(+1)
            right_score = turn_dir_score(-1)
            if left_score == right_score:
                probs = np.array([self.turn_left_weight, self.turn_right_weight], dtype=float)
                probs = probs / probs.sum()
                return a.turn_left if self.rng.random() < probs[0] else a.turn_right
            return a.turn_left if left_score > right_score else a.turn_right

        probs = np.array([self.turn_left_weight, self.turn_right_weight], dtype=float)
        probs = probs / probs.sum()
        return a.turn_left if self.rng.random() < probs[0] else a.turn_right


class CenterRotatePolicy:
    """At each timestep, rotate randomly (left or right), never move forward."""

    def __init__(self, env: gym.Env):
        self.env = env.unwrapped
        self.rng = self.env.np_random

    def action(self, step_idx: int) -> int:
        a = self.env.actions
        return a.turn_left if (self.rng.random() < 0.5) else a.turn_right


def _wrap_angle(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


def run_rollout(
    env: gym.Env,
    steps: int,
    align_heading_zero: bool,
    segment_len: int,
    policy_name: str,
    policy_kwargs: dict,
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
    rgb = env.render()  # render current obs

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

    if policy_name == "back_and_forth":
        policy = BackAndForthPolicy(segment_len=segment_len)
        act_fn = policy.action
    elif policy_name == "center_rotate":
        policy = CenterRotatePolicy(env=env)
        act_fn = policy.action
    else:
        policy = BiasedRandomPolicy(env=env, **policy_kwargs)
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
    from tqdm import tqdm

    assert args.dataset_root and args.num_videos > 0, "Provide --dataset-root and --num-videos"
    ds_root = Path(args.dataset_root)
    ds_root.mkdir(parents=True, exist_ok=True)

    ns = SimpleNamespace(
        args=args,
        dataset_root=str(ds_root),
        block_size=int(args.block_size),
        file_digits=int(args.file_digits),
    )

    indices = list(range(int(args.num_videos)))

    with ProcessPoolExecutor(max_workers=int(args.num_processes)) as ex:
        futures = {ex.submit(_generate_one, idx, ns): idx for idx in indices}
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Generating videos"):
            pass


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
    parser.add_argument("--grid-mode", action="store_true")
    parser.add_argument("--grid-vel-min", type=int, default=-1)
    parser.add_argument("--grid-vel-max", type=int, default=1)
    parser.add_argument("--debug-join", dest="debug_join", action="store_true", help="save a side-by-side debug video with RGB (left) and top-view map (right)")
    parser.add_argument("--output-2d-map", dest="output_2d_map", action="store_true", help="save the top-view map as a separate MP4 named *_map_2d.mp4")
    parser.add_argument("--spawn-wall-buffer", type=float, default=1.0, help="extra buffer from walls when spawning agent and boxes (meters)")
    parser.add_argument("--room-size", type=int, default=12, help="square room side length in meters (e.g., 12)")
    parser.add_argument("--even-lighting", action="store_true", help="use uniform ambient lighting (no directional shading)")
    parser.add_argument("--floor-tex", type=str, default="concrete", help="floor texture name (see miniworld/textures), default white")
    # parser.add_argument("--floor-tex", type=str, default="ceiling_tile_noborder", help="floor texture name (see miniworld/textures), default white")
    parser.add_argument("--wall-tex", type=str, default="white", help="wall texture name, default white")
    parser.add_argument("--ceil-tex", type=str, default="ceiling_tile_noborder", help="ceiling texture name, default white")
    # parser.add_argument("--ceil-tex", type=str, default="wood", help="ceiling texture name, default white")
    # Block size controls (MovingBlockWorld): uniform footprint and/or height
    parser.add_argument("--block-size-xy", type=float, default=None, help="if set, use this x/z size (meters) for all blocks")
    parser.add_argument("--block-height", type=float, default=None, help="if set, use this height (meters) for all blocks")

    # Parallel dataset generation
    parser.add_argument("--dataset-root", type=str, default=None, help="if set, run in parallel dataset generation mode and write outputs under this root")
    parser.add_argument("--num-videos", type=int, default=0, help="total number of videos to generate (parallel mode)")
    parser.add_argument("--block-size", type=int, default=100, help="number of videos per block subdirectory (parallel mode)")
    parser.add_argument("--num-processes", type=int, default=4, help="number of worker processes (parallel mode)")
    parser.add_argument("--file-digits", type=int, default=4, help="zero-padding width for filenames inside each block (parallel mode)")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="base seed for deterministic generation across items; if omitted, each item uses a random seed",
    )

    # Policy selection and knobs
    parser.add_argument("--policy", type=str, default="biased_random", choices=["biased_random", "back_and_forth", "center_rotate"], help="which control policy to use")
    parser.add_argument("--forward-prob", type=float, default=0.8, help="biased_random: probability of moving forward when safe")
    parser.add_argument("--turn-left-weight", type=float, default=1.0, help="biased_random: relative weight for choosing left turns")
    parser.add_argument("--turn-right-weight", type=float, default=1.0, help="biased_random: relative weight for choosing right turns")
    parser.add_argument("--wall-buffer", type=float, default=1.5, help="biased_random: minimum distance from walls before turning away")
    parser.add_argument("--avoid-turning-into-walls", action="store_true", help="biased_random: when turning, prefer direction that increases distance to walls")
    parser.add_argument("--lookahead-mult", type=float, default=2.0, help="biased_random: lookahead distance multiplier relative to max forward step")
    # Agent spawn option: place agent at the center (uses env support)
    parser.add_argument("--agent-center-start", action="store_true", help="spawn the agent at the room center (top-left of middle for even sizes)")

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


