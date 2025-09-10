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
  --dataset-root /data/hansen/wm-memory/data/blockworld/static_center_rotate_validation \
  --num-videos 1000 --block-size 64 --num-processes 32 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 90
      
python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /data/hansen/wm-memory/data/blockworld/static_center_rotate_training \
  --num-videos 20000 --block-size 256 --num-processes 32 \
  -- \
  --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap --box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -0 --grid-vel-max 0 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 10 --no-time-limit --output-2d-map \
  --blocks-static --block-size-xy 0.7 --block-height 1.5 --agent-center-start --policy center_rotate --cam-fov-y 90

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


