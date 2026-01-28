#!/usr/bin/env python3

"""
scripts.generate_videos_batch

Batch launcher for dataset-style generation using subprocess parallelism.

This script launches many *independent* single-rollout jobs by invoking
`python -m scripts.generate_videos` in parallel subprocesses. It replaces the
deprecated in-process parallel mode previously embedded in generate_videos.

Directory / naming scheme:
  Outputs are written under --dataset-root in block subfolders to avoid huge
  directory fanout:
    <dataset-root>/<block_id>/<item_id>_rgb.mp4
    <dataset-root>/<block_id>/<item_id>_depth.pt
    <dataset-root>/<block_id>/<item_id>_actions.pt
  where:
    block_id = idx // --block-size
    item_id  = idx %  --block-size (zero-padded to --file-digits)

Basic usage:
  python -m scripts.generate_videos_batch \
    --env-name MiniWorld-MovingBlockWorld-v0 \
    --dataset-root ./out/multi \
    --num-videos 1000 --block-size 100 --num-processes 8 \
    -- \
    --steps 300 --room-size 10 --policy center_rotate --agent-center-start

Forwarding convention:
  All arguments after the positional "--" separator are forwarded verbatim to
  `scripts.generate_videos` (single-run). This is the recommended place to set
  rollout length, policy, rendering resolution, textures, map/debug outputs, etc.

Key flags:
  --env-name STR            Environment id passed to each subprocess.
  --dataset-root PATH       Root output directory (created if missing).
  --num-videos INT          Number of rollouts to generate.
  --block-size INT          Items per block directory (controls subfoldering).
  --file-digits INT         Zero-pad width for per-item filenames (default: 4).
  --num-processes INT       Max concurrent subprocesses (thread pool controls launch).
  --python PATH             Python executable used for subprocesses (default: current).

Execution model:
  Uses a ThreadPoolExecutor to launch and wait on subprocesses (I/O-bound wait).
  Captures stdout/stderr for each job; nonzero exit codes are collected and
  summarized at the end (prints first few failures).

Headless rendering:
  Subprocesses inherit the parent environment. To force headless rendering on
  servers, set:
    MINIWORLD_HEADLESS=1 python -m scripts.generate_videos_batch ...

See also:
  scripts.generate_videos  (single rollout + outputs + policy/env knobs)
  

Generate videos example script for including block info
IMPORTANT: ONLY 6 blocks, one of each color.
140 steps only, so only one possible clip per video
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/sunday_v2_block_info_validation_6_blocks_v2 \
  --num-videos 1024 --block-size 64 --num-processes 32 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 140 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 6 --ensure-base-palette --store-block-info
  
  
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
    parser.add_argument(
        "--store-block-info",
        action="store_true",
        help="also write <out-prefix>_block_info.pt for each rollout (same as scripts.generate_videos --store-block-info)",
    )
    # Forward the rest of args verbatim to scripts.generate_videos
    parser.add_argument("rest", nargs=argparse.REMAINDER, help="arguments to forward to scripts.generate_videos (must be after --)")

    args = parser.parse_args()

    # Clean up forwarded args: drop a leading "--" if present
    fwd = list(args.rest)
    if fwd and fwd[0] == "--":
        fwd = fwd[1:]
    # If requested, ensure the underlying single-run script gets the flag
    if args.store_block_info and ("--store-block-info" not in fwd):
        fwd = fwd + ["--store-block-info"]

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


