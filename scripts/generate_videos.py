#!/usr/bin/env python3

"""
Generate rollout videos (RGB and depth) and actions for a simple agent policy.

Example usage (mirrors manual_control flags):

python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlocksWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --box-allow-overlap --agent-box-allow-overlap \
  --box-random-orientation \
  --steps 300 --out-prefix ./out/run1

Outputs:
  - <out-prefix>_rgb.mp4
  - <out-prefix>_depth.mp4
  - <out-prefix>_actions.pt
"""

import argparse
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

import miniworld
from miniworld.params import DEFAULT_PARAMS


def build_env(args) -> gym.Env:
    view_mode = "top" if args.top_view else "agent"
    env_kwargs = {"view": view_mode, "render_mode": "rgb_array"}

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

    # Build params so first reset uses them
    if args.turn_step_deg is not None or args.forward_step is not None:
        params = DEFAULT_PARAMS.no_random()
        if args.turn_step_deg is not None:
            v = float(args.turn_step_deg)
            params.set("turn_step", v, v, v)
        if args.forward_step is not None:
            v = float(args.forward_step)
            params.set("forward_step", v, v, v)
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


def simple_back_and_forth_policy(step_idx: int, period: int = 40) -> int:
    """Toggle move_forward / move_back every half-period."""
    half = period // 2
    return 2 if (step_idx % period) < half else 3  # MiniWorldEnv.Actions indices


def run_rollout(env: gym.Env, steps: int, align_heading_zero: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs_list = []
    depth_list = []
    actions = []

    obs, _ = env.reset()
    if align_heading_zero:
        env.unwrapped.agent.dir = 0.0
    rgb = env.render()  # render current obs

    # Collect first frame
    obs_list.append(rgb)
    depth = env.unwrapped.render_depth()
    depth_list.append(depth)

    for t in range(steps):
        a = simple_back_and_forth_policy(t)
        obs, reward, term, trunc, info = env.step(a)
        rgb = env.render()
        obs_list.append(rgb)
        depth = env.unwrapped.render_depth()
        depth_list.append(depth)
        actions.append(a)

        if term or trunc:
            break

    # Stack to arrays
    rgb_arr = np.stack(obs_list, axis=0)  # T,H,W,3 uint8
    depth_arr = np.stack(depth_list, axis=0)  # T,H,W,1 float32
    actions_arr = np.array(actions, dtype=np.int64)
    return rgb_arr, depth_arr, actions_arr


def write_mp4_rgb(out_path: str, frames: np.ndarray, fps: int = 30):
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


def write_mp4_depth(out_path: str, depth_frames: np.ndarray, fps: int = 30):
    import imageio

    # Normalize depth to 0..255 for visualization; keep as grayscale
    d = depth_frames
    # Clip extreme far distances for visibility
    d = np.clip(d, 0.0, np.percentile(d, 99.5))
    d = d / (d.max() + 1e-6)
    d_uint8 = (d * 255).astype(np.uint8)
    d_uint8 = d_uint8.squeeze(-1)  # T,H,W
    out_dir = os.path.dirname(out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    with imageio.get_writer(
        out_path,
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        bitrate="8M",
    ) as writer:
        for frame in d_uint8:
            writer.append_data(frame)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out-prefix", type=str, required=True)
    parser.add_argument("--top_view", action="store_true")
    parser.add_argument("--domain-rand", action="store_true")
    parser.add_argument("--no-time-limit", action="store_true")
    parser.add_argument("--heading-zero", action="store_true")

    # Movement/env customization flags (mirror manual_control)
    parser.add_argument("--turn-step-deg", type=float, default=None)
    parser.add_argument("--forward-step", type=float, default=None)
    parser.add_argument("--box-speed-scale", type=float, default=None)
    parser.add_argument("--box-allow-overlap", action="store_true")
    parser.add_argument("--agent-box-allow-overlap", action="store_true")
    parser.add_argument("--box-random-orientation", action="store_true")
    parser.add_argument("--grid-mode", action="store_true")
    parser.add_argument("--grid-vel-min", type=int, default=-1)
    parser.add_argument("--grid-vel-max", type=int, default=1)

    args = parser.parse_args()

    env = build_env(args)
    print(f"Miniworld v{miniworld.__version__}, Env: {args.env_name}")

    rgb, depth, actions = run_rollout(env, args.steps, align_heading_zero=args.heading_zero)

    # Save outputs
    write_mp4_rgb(f"{args.out_prefix}_rgb.mp4", rgb)
    write_mp4_depth(f"{args.out_prefix}_depth.mp4", depth)
    torch.save(torch.tensor(actions, dtype=torch.long), f"{args.out_prefix}_actions.pt")

    env.close()


if __name__ == "__main__":
    main()


