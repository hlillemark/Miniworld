#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
import torch
import imageio.v2 as imageio
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


"""
Process dataset mode
python /data/hansen/projects/Miniworld/scripts/canonicalize_map_fov.py \
  --dataset_root /data/hansen/projects/wm-memory/data/blockworld/static_training_w_map \
  --output_suffix canonicalized_cone_map_2d.mp4 \
  --margin_bottom 8 \
  --center_mode left \
  --save_mask_orig --save_mask_canon --save_masked_map \
  --num_workers 32

Preview mode 
python /data/hansen/projects/Miniworld/scripts/canonicalize_map_fov.py \
  --actions /data/hansen/projects/Miniworld/out/run_move_actions.pt \
  --map_mp4 /data/hansen/projects/Miniworld/out/run_move_map_2d.mp4 \
  --agent_mp4 /data/hansen/projects/Miniworld/out/run_move_rgb.mp4 \
  --outdir /data/hansen/projects/Miniworld/out/2d_map_outputs \
  --margin_bottom 8 --save_stride 8 --center_mode left

"""


# -------------------------
# Utils
# -------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def pil_resize_to(img_np: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Resize HxWxC uint8 np array to target (H, W) with PIL (bilinear)."""
    Ht, Wt = target_hw
    if img_np.shape[0] == Ht and img_np.shape[1] == Wt:
        return img_np
    img = Image.fromarray(img_np)
    img = img.resize((Wt, Ht), resample=Image.BILINEAR)
    return np.asarray(img)


def save_side_by_side(path: Path, imgs: list[np.ndarray], titles: list[str] | None = None):
    """
    Save images horizontally stacked. All images are resized to the smallest H,W among them.
    imgs: list of HxWxC uint8
    """
    assert len(imgs) >= 1
    # make all 3-channel
    normed = []
    for im in imgs:
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=2)
        elif im.shape[2] == 4:
            im = im[..., :3]
        normed.append(im.astype(np.uint8))

    Hmin = min(im.shape[0] for im in normed)
    Wmin = min(im.shape[1] for im in normed)
    resized = [pil_resize_to(im, (Hmin, Wmin)) for im in normed]
    out = np.concatenate(resized, axis=1)
    imageio.imwrite(path, out)


# -------------------------
# Geometry helpers (unchanged conventions)
# -------------------------

# Single world point -> pixel
def world_to_pixel(x_world, z_world, top_view_scale):
    x_scale = float(top_view_scale["x_scale"])
    z_scale = float(top_view_scale["z_scale"])
    x_offset = float(top_view_scale["x_offset"])
    z_offset = float(top_view_scale["z_offset"])
    u = x_world * x_scale + x_offset  # column
    v = z_world * z_scale + z_offset  # row
    return u, v


def fov_cone_mask(
    H, W,
    agent_xy_px,
    heading_rad,
    fov_deg=60.0,
    max_range_px=None,
    back_pixels=8.0,
):
    """
    Boolean FOV mask for a single frame using agent position in *pixel* coords.
    The cone origin is shifted backwards along the opposite of the view direction.

    NOTE: follows your dataset convention exactly (including the sin(-heading_rad) term).
    """
    ax, ay = agent_xy_px

    # Shift backwards from the agent along the heading
    ax_shifted = ax - back_pixels * np.cos(heading_rad)
    ay_shifted = ay - back_pixels * np.sin(-heading_rad)

    xs = np.arange(W, dtype=np.float32)
    ys = np.arange(H, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys)  # (H,W)

    dx = X - ax_shifted
    dy = Y - ay_shifted

    # Rotate coords so forward is +x
    ch = np.cos(heading_rad)
    sh = np.sin(heading_rad)
    dx_f = ch * dx - sh * dy
    dy_f = sh * dx + ch * dy

    in_front = dx_f >= 0.0
    half_fov = np.deg2rad(fov_deg) * 0.5
    ang = np.arctan2(dy_f, dx_f)
    within_angle = np.abs(ang) <= half_fov

    if max_range_px is not None:
        r2 = dx_f * dx_f + dy_f * dy_f
        within_range = r2 <= (max_range_px * max_range_px)
    else:
        within_range = np.ones_like(in_front, dtype=bool)

    return in_front & within_angle & within_range


def apply_fov_mask_to_video_world(
    video,                 # [T,H,W,C]
    agent_positions_w,     # [T,3] or [3] world coords; use (x,z)
    headings_rad,          # [T] or scalar
    top_view_scale,        # dict describing transform world->pixel
    world_size=16.0,       # (kept for completeness; not used except to hint scaling)
    fov_deg=60.0,
    max_range_world=None,  # in world units; converts to pixels
    outside_fill=0         # scalar or (C,) per-channel value
):
    """
    Builds and applies an FOV cone mask per frame using world coords.
    Returns (masked_video, masks) where masks[t] is True inside FOV.
    """
    T, H, W, C = video.shape
    out = video.copy()

    # Normalize positions to (x_world, z_world) per-frame
    pos = np.asarray(agent_positions_w, dtype=np.float32)
    if pos.ndim == 1:
        if pos.shape[-1] != 3:
            raise ValueError("agent_positions_w must be length-3 if 1D (x,y,z).")
        xw = np.full(T, pos[0], dtype=np.float32)
        yw = np.full(T, pos[2], dtype=np.float32)  # z -> image y
    else:
        if pos.shape[-1] != 3 or pos.shape[0] != T:
            raise ValueError("agent_positions_w must be [T,3].")
        xw = pos[:, 0].astype(np.float32)
        yw = pos[:, 2].astype(np.float32)  # z -> image y

    # Headings per-frame
    headings = np.asarray(headings_rad, dtype=np.float32)
    if headings.ndim == 0:
        headings = np.full(T, headings, dtype=np.float32)
    elif headings.shape[0] != T:
        raise ValueError("headings_rad must be scalar or length T.")

    # Convert optional range to pixels using scale hints (if you want isotropic range)
    if max_range_world is not None:
        # Derive average pixel scale from top_view_scale if available;
        # otherwise fall back to W/world_size, H/world_size average.
        if all(k in top_view_scale for k in ("x_scale", "z_scale")):
            sx = float(top_view_scale["x_scale"])
            sy = float(top_view_scale["z_scale"])
        else:
            sx = W / float(world_size)
            sy = H / float(world_size)
        max_range_px = max_range_world * 0.5 * (sx + sy)
    else:
        max_range_px = None

    masks = np.zeros((T, H, W), dtype=bool)

    # Prepare outside fill
    if np.isscalar(outside_fill):
        fill_val = np.array(outside_fill, dtype=out.dtype)
    else:
        fill_val = np.asarray(outside_fill, dtype=out.dtype)
        if fill_val.shape != (C,):
            raise ValueError(f"outside_fill must be scalar or shape ({C},)")

    for t in range(T):
        ax_px, ay_px = world_to_pixel(xw[t], yw[t], top_view_scale)
        mask = fov_cone_mask(
            H, W,
            agent_xy_px=(ax_px, ay_px),
            heading_rad=headings[t],
            fov_deg=fov_deg,
            max_range_px=max_range_px,
            back_pixels= H / 32  # e.g., 256->8 px; 128->4 px
        )
        masks[t] = mask
        if fill_val.ndim == 0:
            out[t][~mask] = fill_val
        else:
            out[t][~mask] = fill_val[None, None, :]

    return out, masks


# 90° rotation & integer-translation canonicalizer (world-anchored)
def _rot_k_from_heading(heading_rad: float) -> int:
    # Dataset: 0→right, +π/2→up, +π→left, +3π/2→down.
    # Return k for np.rot90 so the agent faces UP after rotation.
    n = int(np.round(heading_rad / (np.pi / 2.0))) % 4
    k = (1 - n) % 4
    return k


def _rotate_xy(ax: float, ay: float, H: int, W: int, k: int):
    # Map (ax,ay) under np.rot90(img, k) with axes=(0,1)
    if k == 0:
        return ax, ay
    elif k == 1:  # 90° CCW: (H,W)->(W,H)
        return ay, (W - 1) - ax
    elif k == 2:  # 180°
        return (W - 1) - ax, (H - 1) - ay
    elif k == 3:  # 270° CCW: (H,W)->(W,H)
        return (H - 1) - ay, ax
    else:
        raise ValueError("k must be 0..3")


def _translate_integer(img, tx: int, ty: int, background=0):
    """
    Translate image by integer (tx, ty) with no interpolation.
    +tx => shift content right, +ty => shift content down.
    Implemented as background-filled output + rectangle blit.
    Works for [H,W] or [H,W,C].
    """
    H, W = img.shape[:2]
    out = np.empty_like(img)

    # Fill background
    if img.ndim == 3 and not np.isscalar(background):
        bg = np.asarray(background, dtype=img.dtype)
        if bg.shape != (img.shape[2],):
            raise ValueError(f"background must be scalar or shape ({img.shape[2]},)")
        out[...] = bg
    else:
        out[...] = background

    # Compute overlapping rectangles
    src_x0 = max(0, -tx)
    src_y0 = max(0, -ty)
    dst_x0 = max(0,  tx)
    dst_y0 = max(0,  ty)

    w = min(W - src_x0, W - dst_x0)
    h = min(H - src_y0, H - dst_y0)

    if w > 0 and h > 0:
        if img.ndim == 2:
            out[dst_y0:dst_y0+h, dst_x0:dst_x0+w] = img[src_y0:src_y0+h, src_x0:src_x0+w]
        else:
            out[dst_y0:dst_y0+h, dst_x0:dst_x0+w, :] = img[src_y0:src_y0+h, src_x0:src_x0+w, :]

    return out


def canonicalize_video_discrete_world(
    video,               # [T,H,W,C] or [T,H,W]
    agent_positions_w,   # [T,3] or [3] world coords (use x,z)
    headings_rad,        # [T] or scalar (multiples of π/2)
    top_view_scale,      # dict with x_scale,z_scale,x_offset,z_offset
    margin_bottom=8,     # pixels above bottom
    background=0,        # scalar or (C,)
    center_mode="left",  # "left" or "right" center for even W
    add_half_cell=False  # True if agent marker is drawn at cell centers
):
    """
    Rotate by 90° so agent faces UP, then translate so agent is exactly
    bottom-center (8 px above bottom). No interpolation anywhere.
    """
    T, H, W = video.shape[:3]
    out = np.empty_like(video)

    pos = np.asarray(agent_positions_w, dtype=np.float32)
    if pos.ndim == 1:
        pos = np.tile(pos[None, :], (T, 1))
    heads = np.asarray(headings_rad, dtype=np.float32)
    if heads.ndim == 0:
        heads = np.full((T,), float(heads), dtype=np.float32)

    half = 0.5 if add_half_cell else 0.0

    for t in range(T):
        # 1) world -> pixel
        ax_px, ay_px = world_to_pixel(pos[t, 0] + half, pos[t, 2] + half, top_view_scale)

        # 2) rotate frame to face up (no resampling)
        k = _rot_k_from_heading(heads[t])
        rot = np.rot90(video[t], k=k, axes=(0, 1))
        Hr, Wr = rot.shape[:2]

        # rotate agent coord
        ax_r, ay_r = _rotate_xy(ax_px, ay_px, H, W, k)

        # 3) bottom-center target
        cx_i = (Wr - 1) // 2 if center_mode == "left" else (Wr // 2)
        cy_i = (Hr - 1) - int(margin_bottom)

        # 4) integer shift to land agent exactly at (cx_i, cy_i)
        ax_i = int(np.round(ax_r))
        ay_i = int(np.round(ay_r))
        tx = cx_i - ax_i   # +tx moves content right
        ty = cy_i - ay_i   # +ty moves content down

        out[t] = _translate_integer(rot, tx, ty, background=background)

    return out


# -------------------------
# Dataset processing helpers
# -------------------------

def _get_video_fps(mp4_path: str | Path, default_fps: float = 30.0) -> float:
    try:
        reader = imageio.get_reader(mp4_path)
        meta = reader.get_meta_data()
        fps = float(meta.get("fps", default_fps))
        try:
            reader.close()
        except Exception:
            pass
        return fps
    except Exception:
        return float(default_fps)


def _write_video_mp4(frames: np.ndarray, out_path: Path, fps: float):
    """Write [T,H,W,C] uint8 frames to an MP4 file."""
    ensure_dir(out_path.parent)
    writer = imageio.get_writer(out_path, fps=fps)
    try:
        for t in range(frames.shape[0]):
            writer.append_data(frames[t])
    finally:
        try:
            writer.close()
        except Exception:
            pass


def process_sequence_to_canonicalized(actions_path: Path, map_mp4_path: Path, out_mp4_path: Path | None,
                                     fov_deg: float, margin_bottom: int, center_mode: str,
                                     background_value,
                                     out_mask_orig_path: Path | None = None,
                                     out_mask_canon_path: Path | None = None,
                                     out_masked_map_path: Path | None = None):
    """
    Load one sequence, build masked + canonicalized map frames, and write to out_mp4_path.
    """
    actions = torch.load(actions_path)
    agent_pos_world = actions["agent_pos"]
    agent_view_dir = actions["agent_dir"]
    top_view_scale = actions["top_view_scale"]

    map_vid = np.stack(imageio.mimread(map_mp4_path), axis=0)  # [T,H,W,C]

    masked_map_vid, masks = apply_fov_mask_to_video_world(
        map_vid,
        agent_positions_w=agent_pos_world,
        headings_rad=agent_view_dir,
        top_view_scale=top_view_scale,
        world_size=16.0,
        fov_deg=fov_deg,
        max_range_world=None,
        outside_fill=0
    )

    canonicalized_map_vid = canonicalize_video_discrete_world(
        masked_map_vid,
        agent_positions_w=agent_pos_world,
        headings_rad=agent_view_dir,
        top_view_scale=top_view_scale,
        margin_bottom=margin_bottom,
        background=background_value,
        center_mode=center_mode,
        add_half_cell=False
    )

    fps = _get_video_fps(map_mp4_path, default_fps=30.0)
    if out_mp4_path is not None:
        _write_video_mp4(canonicalized_map_vid.astype(np.uint8), out_mp4_path, fps=fps)

    # Optional extra outputs (dataset-mode)
    if out_mask_orig_path is not None or out_mask_canon_path is not None or out_masked_map_path is not None:
        # (T,H,W) uint8 masks (0 or 255)
        mask_u8 = (masks.astype(np.uint8) * 255)

        if out_mask_orig_path is not None:
            _write_video_mp4(mask_u8.astype(np.uint8), out_mask_orig_path, fps=fps)

        if out_mask_canon_path is not None:
            canon_mask_u8 = canonicalize_video_discrete_world(
                mask_u8,
                agent_positions_w=agent_pos_world,
                headings_rad=agent_view_dir,
                top_view_scale=top_view_scale,
                margin_bottom=margin_bottom,
                background=0,
                center_mode=center_mode,
                add_half_cell=False
            )
            _write_video_mp4(canon_mask_u8.astype(np.uint8), out_mask_canon_path, fps=fps)

        if out_masked_map_path is not None:
            _write_video_mp4(masked_map_vid.astype(np.uint8), out_masked_map_path, fps=fps)


def _run_dataset_job(actions_path: Path,
                     map_mp4_path: Path,
                     out_mp4_path: Path | None,
                     fov_deg: float,
                     margin_bottom: int,
                     center_mode: str,
                     background_value,
                     out_mask_orig_path: Path | None,
                     out_mask_canon_path: Path | None,
                     out_masked_map_path: Path | None):
    try:
        process_sequence_to_canonicalized(
            actions_path=actions_path,
            map_mp4_path=map_mp4_path,
            out_mp4_path=out_mp4_path,
            fov_deg=fov_deg,
            margin_bottom=margin_bottom,
            center_mode=center_mode,
            background_value=background_value,
            out_mask_orig_path=out_mask_orig_path,
            out_mask_canon_path=out_mask_canon_path,
            out_masked_map_path=out_masked_map_path,
        )
        return True, str(actions_path)
    except Exception as e:
        return False, f"{actions_path}: {e}"


# -------------------------
# Main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build masked & canonicalized map frames and save preview PNGs.")
    p.add_argument("--actions", type=str, default="/data/hansen/projects/Miniworld/out/run_move_actions.pt")
    p.add_argument("--map_mp4", type=str, default="/data/hansen/projects/Miniworld/out/run_move_map_2d.mp4")
    p.add_argument("--agent_mp4", type=str, default="/data/hansen/projects/Miniworld/out/run_move_rgb.mp4")
    p.add_argument("--outdir", type=str, default="./out/2d_map_outputs/")
    p.add_argument("--fov_deg", type=float, default=60.0)
    p.add_argument("--margin_bottom", type=int, default=8)
    p.add_argument("--max_frames", type=int, default=100, help="Process/save at most this many frames (in order).")
    p.add_argument("--save_stride", type=int, default=8, help="Save every Nth frame for previews.")
    p.add_argument("--center_mode", type=str, choices=["left", "right"], default="left")
    p.add_argument("--background", type=int, nargs="+", default=[0], help="Background value (scalar or 3 ints).")
    # Dataset mode
    p.add_argument("--dataset_root", type=str, default=None, help="If set, process dataset directory instead of single-run previews.")
    p.add_argument("--dataset_subdir", type=str, default=None, help="Optional subdirectory under dataset_root. If omitted, search recursively under dataset_root.")
    p.add_argument("--output_suffix", type=str, default="canonicalized_cone_map_2d.mp4", help="Suffix for output video filenames.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs in dataset mode.")
    # Extra outputs toggles (dataset mode)
    p.add_argument("--save_mask_orig", action="store_true", help="Also save the original FOV mask (1-channel) as mp4 next to sequence.")
    p.add_argument("--save_mask_canon", action="store_true", help="Also save the canonicalized FOV mask (1-channel) as mp4 next to sequence.")
    p.add_argument("--save_masked_map", action="store_true", help="Also save the non-canonicalized masked 2D map (3-channel) next to sequence.")
    p.add_argument("--num_workers", type=int, default=1, help="Number of worker processes for dataset mode.")
    return p.parse_args()


def main():
    args = parse_args()
    # Determine background for canonicalization
    background = args.background[0] if len(args.background) == 1 else np.array(args.background, dtype=np.uint8)

    # Dataset processing mode
    if args.dataset_root is not None and len(str(args.dataset_root)) > 0:
        root = Path(args.dataset_root)
        search_base = root / args.dataset_subdir if args.dataset_subdir else root
        actions_files = sorted(search_base.rglob("*_actions.pt"))
        if not actions_files:
            print(f"No actions files found under {search_base}")
            return
        print(f"Found {len(actions_files)} sequences under {search_base}")
        tasks = []
        for ap in actions_files:
            idx = ap.stem.replace("_actions", "")
            seq_dir = ap.parent
            map_mp4 = seq_dir / f"{idx}_map_2d.mp4"
            if not map_mp4.exists():
                print(f"[WARN] Missing map video for {idx} -> {map_mp4}")
                continue
            out_mp4 = seq_dir / f"{idx}_{args.output_suffix}"
            out_mask_orig = seq_dir / f"{idx}_cone_mask_orig.mp4"
            out_mask_canon = seq_dir / f"{idx}_cone_mask_canon.mp4"
            out_masked_map = seq_dir / f"{idx}_masked_cone_map_2d.mp4"

            # Decide which outputs to generate
            need_canon = args.overwrite or (not out_mp4.exists())
            need_mask_orig = args.save_mask_orig and (args.overwrite or (not out_mask_orig.exists()))
            need_mask_canon = args.save_mask_canon and (args.overwrite or (not out_mask_canon.exists()))
            need_masked_map = args.save_masked_map and (args.overwrite or (not out_masked_map.exists()))

            if not (need_canon or need_mask_orig or need_mask_canon or need_masked_map):
                print(f"[SKIP] All outputs exist for {idx}")
                continue

            tasks.append((ap, map_mp4,
                          out_mp4 if need_canon else None,
                          args.fov_deg,
                          args.margin_bottom,
                          args.center_mode,
                          background,
                          out_mask_orig if need_mask_orig else None,
                          out_mask_canon if need_mask_canon else None,
                          out_masked_map if need_masked_map else None))

        if not tasks:
            return

        if args.num_workers and args.num_workers > 1:
            print(f"Launching {len(tasks)} jobs with {args.num_workers} workers...")
            done = 0
            ok = 0
            with ProcessPoolExecutor(max_workers=int(args.num_workers)) as ex:
                futures = [ex.submit(_run_dataset_job, *t) for t in tasks]
                for fut in as_completed(futures):
                    success, msg = fut.result()
                    done += 1
                    ok += 1 if success else 0
                    status = "OK" if success else "FAIL"
                    print(f"[{status}] ({done}/{len(tasks)}) {msg}")
            print(f"Completed {done} jobs, {ok} succeeded, {done-ok} failed.")
        else:
            print(f"Processing {len(tasks)} jobs serially...")
            done = 0
            ok = 0
            for t in tasks:
                success, msg = _run_dataset_job(*t)
                done += 1
                ok += 1 if success else 0
                status = "OK" if success else "FAIL"
                print(f"[{status}] ({done}/{len(tasks)}) {msg}")
            print(f"Completed {done} jobs, {ok} succeeded, {done-ok} failed.")
        return

    # Single-run preview mode (original functionality)
    outdir = Path(args.outdir)
    ensure_dir(outdir)

    # Load actions
    actions = torch.load(args.actions)
    agent_pos_world = actions["agent_pos"]      # [T,3] or [3]
    agent_view_dir  = actions["agent_dir"]      # [T]   or scalar (radians)
    top_view_scale  = actions["top_view_scale"] # dict

    # Load videos
    map_vid = np.stack(imageio.mimread(args.map_mp4), axis=0)          # [T,H,W,C]
    agent_view_vid = np.stack(imageio.mimread(args.agent_mp4), axis=0) # [T,H,W,C]

    # Truncate to max_frames if requested
    T = map_vid.shape[0]
    if args.max_frames is not None and args.max_frames > 0:
        T = min(T, int(args.max_frames))
        map_vid = map_vid[:T]
        agent_view_vid = agent_view_vid[:T]
        if isinstance(agent_pos_world, np.ndarray):
            agent_pos_world = agent_pos_world[:T]
        else:
            try:
                agent_pos_world = agent_pos_world[:T]
            except Exception:
                pass
        if hasattr(agent_view_dir, "__len__") and not np.isscalar(agent_view_dir):
            agent_view_dir = agent_view_dir[:T]

    # Build masked map
    masked_map_vid, masks = apply_fov_mask_to_video_world(
        map_vid,
        agent_positions_w=agent_pos_world,
        headings_rad=agent_view_dir,
        top_view_scale=top_view_scale,
        world_size=16.0,
        fov_deg=args.fov_deg,
        max_range_world=None,
        outside_fill=0
    )

    # Canonicalize masked map (world-anchored)
    canonicalized_map_vid = canonicalize_video_discrete_world(
        masked_map_vid,
        agent_positions_w=agent_pos_world,
        headings_rad=agent_view_dir,
        top_view_scale=top_view_scale,
        margin_bottom=args.margin_bottom,
        background=background,
        center_mode=args.center_mode,
        add_half_cell=False
    )

    # Save preview frames
    for t in range(0, T, max(1, args.save_stride)):
        sbs_path = outdir / f"preview_t{t:04d}.png"
        save_side_by_side(
            sbs_path,
            [map_vid[t], masked_map_vid[t], canonicalized_map_vid[t]],
            titles=None
        )

    print(f"Saved {len(list(outdir.glob('preview_t*.png')))} previews to {outdir}")


if __name__ == "__main__":
    main()
