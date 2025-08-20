#!/usr/bin/env python3

"""
This script allows you to manually control the simulator
using the keyboard arrows.
"""

import argparse

import gymnasium as gym

import miniworld
from miniworld.params import DEFAULT_PARAMS
from miniworld.manual_control import ManualControl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", default="MiniWorld-Hallway-v0")
    parser.add_argument(
        "--domain-rand", action="store_true", help="enable domain randomization"
    )
    parser.add_argument(
        "--no-time-limit", action="store_true", help="ignore time step limits"
    )
    parser.add_argument(
        "--top_view",
        action="store_true",
        help="show the top view instead of the agent view",
    )
    parser.add_argument(
        "--box-speed-scale",
        type=float,
        default=None,
        help="scale factor for box velocity in PutNextMoving (e.g., 2.0 for 2x)",
    )
    parser.add_argument(
        "--box-allow-overlap",
        action="store_true",
        help="allow moving boxes to pass through each other (only collide with walls/agent)",
    )
    parser.add_argument(
        "--agent-box-allow-overlap",
        action="store_true",
        help="allow agent and boxes to pass through each other",
    )
    parser.add_argument(
        "--box-random-orientation",
        action="store_true",
        help="randomize initial box orientations (otherwise align to 0)",
    )
    parser.add_argument(
        "--grid-mode",
        action="store_true",
        help="snap positions to integer grid and use integer velocities for boxes",
    )
    parser.add_argument(
        "--grid-vel-min",
        type=int,
        default=-1,
        help="min integer velocity component for boxes in grid mode (inclusive)",
    )
    parser.add_argument(
        "--grid-vel-max",
        type=int,
        default=1,
        help="max integer velocity component for boxes in grid mode (inclusive)",
    )
    parser.add_argument(
        "--turn-step-deg",
        type=float,
        default=None,
        help="degrees per turn action (e.g., 90)",
    )
    parser.add_argument(
        "--forward-step",
        type=float,
        default=None,
        help="meters per forward/back action (e.g., 1.0)",
    )
    parser.add_argument(
        "--heading-zero",
        action="store_true",
        help="align agent heading to 0 radians after each reset",
    )
    args = parser.parse_args()
    view_mode = "top" if args.top_view else "agent"

    env_kwargs = {"view": view_mode, "render_mode": "human"}
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

    # If step/turn parameters are provided, build a params object up-front so
    # the very first reset (inside env __init__) picks them up
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
        # Fallback for envs that don't accept box_speed_scale
        if "box_speed_scale" in env_kwargs:
            print(
                "Warning: --box-speed-scale not supported by this env; ignoring."
            )
            env_kwargs.pop("box_speed_scale")
        if "box_allow_overlap" in env_kwargs:
            print(
                "Warning: --box-allow-overlap not supported by this env; ignoring."
            )
            env_kwargs.pop("box_allow_overlap")
        if "agent_box_allow_overlap" in env_kwargs:
            print(
                "Warning: --agent-box-allow-overlap not supported by this env; ignoring."
            )
            env_kwargs.pop("agent_box_allow_overlap")
        if "box_random_orientation" in env_kwargs:
            print(
                "Warning: --box-random-orientation not supported by this env; ignoring."
            )
            env_kwargs.pop("box_random_orientation")
        for k in ["grid_mode", "grid_vel_min", "grid_vel_max"]:
            if k in env_kwargs:
                print(f"Warning: --{k.replace('_','-')} not supported by this env; ignoring.")
                env_kwargs.pop(k)
        env = gym.make(args.env_name, **env_kwargs)
    miniworld_version = miniworld.__version__

    print(f"Miniworld v{miniworld_version}, Env: {args.env_name}")

    manual_control = ManualControl(
        env, args.no_time_limit, args.domain_rand, args.heading_zero
    )
    manual_control.run()


if __name__ == "__main__":
    main()
