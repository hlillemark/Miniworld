#!/usr/bin/env python3

"""
Launch multiple independent single-generation runs of scripts.generate_videos
in parallel subprocesses. This replaces the old in-process parallel mode.

Usage example:

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root ./out/multi \
  --num-videos 1000 --block-size 100 --num-processes 8 \
  -- --steps 300 --room-size 10 --policy center_rotate --agent-center-start \
     --block-size-xy 0.7 --block-height 1.5 --cam-fov-y 90
     
# VERSION FOR static center rotate with worldsize 10 testing:

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/static_center_rotate_fov60_validation \
  --num-videos 1000 --block-size 64 --num-processes 64 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 60
      
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/static_center_rotate_fov60_training \
  --num-videos 20000 --block-size 256 --num-processes 64 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 60

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/static_nomove_training \
  --num-videos 4000 --block-size 256 --num-processes 64 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy do_nothing --cam-fov-y 60
  
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/static_nomove_validation \
  --num-videos 500 --block-size 128 --num-processes 64 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy do_nothing --cam-fov-y 60
  

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/edge_plus_training \
  --num-videos 20000 --block-size 256 --num-processes 64 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --grid-cardinal-only --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 300 --room-size 16 --output-2d-map \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap \
  --policy edge_plus --observe-steps 5 --cam-fov-y 60

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/edge_plus_validation \
  --num-videos 1000 --block-size 64 --num-processes 64 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --grid-cardinal-only --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 300 --room-size 16 --output-2d-map \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap \
  --policy edge_plus --observe-steps 5 --cam-fov-y 60

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/randwalk_v2_training \
  --num-videos 10000 --block-size 256 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.85 --observe-steps 5 --cam-fov-y 60

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/randwalk_v2_validation \
  --num-videos 1000 --block-size 64 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --debug-join --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.85 --observe-steps 5 --cam-fov-y 60


python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/randwalk_og_updated_training \
  --num-videos 10000 --block-size 256 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_random --forward-prob 0.9 --cam-fov-y 60

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/randwalk_og_updated_validation \
  --num-videos 1000 --block-size 64 --num-processes 64 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_random --forward-prob 0.9 --cam-fov-y 60



python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/edge_plus_torus_training \
  --num-videos 10000 --block-size 256 --num-processes 16 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --grid-cardinal-only --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 300 --room-size 16 --output-2d-map \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap \
  --policy edge_plus --observe-steps 5 --cam-fov-y 60 \
  --block-torus-wrap
  
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/edge_plus_torus_validation \
  --num-videos 1000 --block-size 64 --num-processes 16 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --grid-cardinal-only --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 300 --room-size 16 --output-2d-map \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap \
  --policy edge_plus --observe-steps 5 --cam-fov-y 60 \
  --block-torus-wrap
  

# Peekaboo debug


python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/peekaboo_full_training \
  --num-videos 10000 --block-size 256 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy peeakboo --observe-steps 70 --cam-fov-y 60


python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/peekaboo_full_validation \
  --num-videos 1000 --block-size 64 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy peeakboo --observe-steps 70 --cam-fov-y 60
  
  
# SUNDAY V2
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/sunday_v2_training \
  --num-videos 10000 --block-size 256 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/sunday_v2_validation \
  --num-videos 1000 --block-size 64 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette
  
  
  
# TEX
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/tex_training \
  --num-videos 10000 --block-size 256 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/tex_validation \
  --num-videos 1000 --block-size 64 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball

  
# TEX vis for new figure 1
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /Users/hansen/Desktop/ucsd/data/miniworld/tex \
  --num-videos 25 --block-size 1 --num-processes 25 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --output-2d-map --room-size 12 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only --agent-center-start \
  --policy center_rotate --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 2 --num-blocks-max 2 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball


# TEX_BOUNCE
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/tex_bounce_training \
  --num-videos 10000 --block-size 256 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 12 \
  --block-size-xy 0.7 --block-height 1.5 \
  --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 8 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/tex_bounce_validation \
  --num-videos 1000 --block-size 64 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 12 \
  --block-size-xy 0.7 --block-height 1.5 \
  --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 8 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball


  
  
# MOVING PEEKABOO
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/peekaboo_motion_training \
  --num-videos 10000 --block-size 256 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy peekaboo_motion --observe-inward-steps 5 --observe-outward-steps 70 --cam-fov-y 60
  
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/peekaboo_motion_validation \
  --num-videos 1000 --block-size 64 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy peekaboo_motion --observe-inward-steps 5 --observe-outward-steps 70 --cam-fov-y 60
  

# STATIC SUNDAY V2
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/sunday_v2_static_training \
  --num-videos 10000 --block-size 256 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.9 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette --blocks-static

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/projects/wm-memory/data/blockworld/sunday_v2_static_validation \
  --num-videos 1000 --block-size 64 --num-processes 48 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.9 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette --blocks-static

Note the positional "--" separator: all arguments after it are forwarded
verbatim to scripts.generate_videos (single-run).
"""

import argparse
import os
import sys
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm


def _run_one(cmd, env):
    res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(cmd)}\nSTDERR:\n{res.stderr.decode('utf-8', errors='ignore')}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--dataset-root", required=True, help="root directory for outputs; subfolders created per block")
    parser.add_argument("--num-videos", type=int, required=True)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--file-digits", type=int, default=4, help="zero-pad width for filenames inside each block")
    parser.add_argument("--python", default=sys.executable, help="python executable to use for subprocesses")
    # Forward the rest of args verbatim to scripts.generate_videos
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="arguments to forward to scripts.generate_videos (must be after --)")

    args = parser.parse_args()

    # Clean up forwarded args: drop a leading "--" if present
    fwd = list(args.rest)
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]

    ds_root = Path(args.dataset_root)
    ds_root.mkdir(parents=True, exist_ok=True)

    total = int(args.num_videos)
    block_size = int(args.block_size)
    file_digits = int(args.file_digits)

    cmds = []
    for idx in range(total):
        block_id = idx // block_size
        item_id = idx % block_size
        block_dir = ds_root / f"{block_id}"
        block_dir.mkdir(parents=True, exist_ok=True)
        stem = str(item_id).zfill(file_digits)
        out_prefix = str(block_dir / stem)

        # Build command; ensure we do not duplicate out-prefix
        cmd = [
            args.python,
            "-m",
            "scripts.generate_videos",
            "--env-name",
            args.env_name,
            "--out-prefix",
            out_prefix,
        ] + fwd

        cmds.append(cmd)

    # Inherit environment; optionally ensure headless rendering if desired
    env = os.environ.copy()

    # Run with a thread pool that launches subprocesses (I/O bound waiting)
    errors = []
    with ThreadPoolExecutor(max_workers=int(args.num_processes)) as ex:
        futs = [ex.submit(_run_one, cmd, env) for cmd in cmds]
        with tqdm(total=len(futs), desc="Generating videos", unit="vid") as pbar:
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    errors.append(str(e))
                finally:
                    pbar.update(1)

    if errors:
        print(f"Completed with {len(errors)} failures:")
        for msg in errors[:10]:
            print("- ", msg.splitlines()[0])
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more")


if __name__ == "__main__":
    main()


