


#### FROM generate_videos.py ####

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


python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --out-prefix ./out/biased_walk_v2 --debug-join --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.9 --cam-fov-y 60 --forward-prob 0.9
  
peekaboo w motion
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --out-prefix ./out/peekaboo_motion --debug-join --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy peekaboo_motion --observe-inward-steps 7 --observe-outward-steps 28 --cam-fov-y 60
  
  
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --out-prefix ./out/static_v2 --debug-join --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.92 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette --blocks-static


Test with bouncing blocks and different texture on the walls etc

python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 100 --output-2d-map --room-size 12 \
  --block-size-xy 0.7 --block-height 1.5 \
  --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 8 --ensure-base-palette \
  --out-prefix ./out/bounce_tex --debug-join \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball
  
  --wall-tex wood --floor-tex cardboard --randomize-box-tex


# BLOCKMOVER
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 100 --output-2d-map --room-size 12 \
  --block-size-xy 0.7 --block-height 1.5 \
  --grid-cardinal-only --blocks-static \
  --policy blockmover --forward-prob 0.92 --cam-fov-y 60 \
  --num-blocks-min 1 --num-blocks-max 1 --ensure-base-palette \
  --out-prefix ./out/blockmover --debug-join
  
  \
  --wall-tex wood --floor-tex cardboard


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



#### from generate_videos_batch.py ####

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
