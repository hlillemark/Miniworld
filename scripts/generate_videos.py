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
  
  
command 7pm aug 20
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlocksWorld-v0 \
  --policy biased_random --forward-prob 0.90 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --render-width 256 --render-height 256 --obs-width 128 --obs-height 128 \
  --steps 300 --out-prefix ./out/run_move --debug --room-size 16
  
"""

import argparse
import math
import os
from typing import Tuple

import gymnasium as gym
import numpy as np
import torch

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


def run_rollout(
    env: gym.Env,
    steps: int,
    align_heading_zero: bool,
    segment_len: int,
    policy_name: str,
    policy_kwargs: dict,
    capture_top: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    obs_list = []
    depth_list = []
    top_list = [] if capture_top else None
    actions = []

    obs, _ = env.reset()
    if align_heading_zero:
        env.unwrapped.agent.dir = 0.0
    rgb = env.render()  # render current obs

    # Collect first frame
    obs_list.append(rgb)
    depth = env.unwrapped.render_depth(env.unwrapped.vis_fb)
    depth_list.append(depth)
    if capture_top:
        top = env.unwrapped.render_top_view(env.unwrapped.vis_fb, render_agent=True)
        top_list.append(top)

    if policy_name == "back_and_forth":
        policy = BackAndForthPolicy(segment_len=segment_len)
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

        if term or trunc:
            break

    # Stack to arrays
    rgb_arr = np.stack(obs_list, axis=0)  # T,H,W,3 uint8
    depth_arr = np.stack(depth_list, axis=0)  # T,H,W,1 float32
    actions_arr = np.array(actions, dtype=np.int64)
    top_arr = np.stack(top_list, axis=0) if capture_top else None
    return rgb_arr, depth_arr, actions_arr, top_arr


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


def write_mp4_depth(out_path: str, depth_frames: np.ndarray, fps: int = 15):
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
    parser.add_argument("--grid-mode", action="store_true")
    parser.add_argument("--grid-vel-min", type=int, default=-1)
    parser.add_argument("--grid-vel-max", type=int, default=1)
    parser.add_argument("--debug", action="store_true", help="save a side-by-side debug video with RGB (left) and top-view map (right)")
    parser.add_argument("--room-size", type=int, default=None, help="square room side length in meters (e.g., 12)")

    # Policy selection and knobs
    parser.add_argument("--policy", type=str, default="biased_random", choices=["biased_random", "back_and_forth"], help="which control policy to use")
    parser.add_argument("--forward-prob", type=float, default=0.8, help="biased_random: probability of moving forward when safe")
    parser.add_argument("--turn-left-weight", type=float, default=1.0, help="biased_random: relative weight for choosing left turns")
    parser.add_argument("--turn-right-weight", type=float, default=1.0, help="biased_random: relative weight for choosing right turns")
    parser.add_argument("--wall-buffer", type=float, default=1.5, help="biased_random: minimum distance from walls before turning away")
    parser.add_argument("--avoid-turning-into-walls", action="store_true", help="biased_random: when turning, prefer direction that increases distance to walls")
    parser.add_argument("--lookahead-mult", type=float, default=2.0, help="biased_random: lookahead distance multiplier relative to max forward step")

    args = parser.parse_args()

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

    rgb, depth, actions, top = run_rollout(
        env,
        args.steps,
        align_heading_zero=args.heading_zero,
        segment_len=args.segment_len,
        policy_name=args.policy,
        policy_kwargs=policy_kwargs,
        capture_top=args.debug,
    )

    # Save outputs
    write_mp4_rgb(f"{args.out_prefix}_rgb.mp4", rgb)
    write_mp4_depth(f"{args.out_prefix}_depth.mp4", depth)
    torch.save(torch.tensor(actions, dtype=torch.long), f"{args.out_prefix}_actions.pt")

    if args.debug and top is not None:
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

    env.close()


if __name__ == "__main__":
    main()


