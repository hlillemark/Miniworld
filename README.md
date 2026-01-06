<p align="center">
    <img src="https://raw.githubusercontent.com/Farama-Foundation/Miniworld/master/miniworld-text.png" width="500px"/>
</p>

**See the instructions after the Installation for the [3D Dynamic Blockworld dataset generation instructions](#3d-dynamic-blockworld-usage), used in the paper [Flow Equivariant World Models](https://flowequivariantworldmodels.github.io). This repository is based off Miniworld, see the link [here](https://github.com/Farama-Foundation/Miniworld) for more information on the original repo.**

Contents:
- [Introduction](#introduction)
- [Installation](#installation)
- [3D Dynamic Blockworld Generation](#3d-dynamic-blockworld-usage)
- [Original Miniworld Usage](#original-miniworld-usage)
- [Other Miniworld Environments](https://miniworld.farama.org/content/env_list/)
- [Miniworld Design and Customization](https://miniworld.farama.org/content/design/)
- [Troubleshooting](https://miniworld.farama.org/content/troubleshooting/)

<!-- ## Introduction

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement
learning &amp; robotics research. It can be used to simulate environments with
rooms, doors, hallways and various objects (eg: office and home environments, mazes).
MiniWorld can be seen as a simpler alternative to VizDoom or DMLab. It is written
100% in Python and designed to be easily modified or extended by students.

<p align="center">
    <img src="images/maze_top_view.jpg" width=260 alt="Figure of Maze environment from top view">
    <img src="images/sidewalk_0.jpg" width=260 alt="Figure of Sidewalk environment">
    <img src="images/collecthealth_0.jpg" width=260 alt="Figure of Collect Health environment">
</p> -->

## Introduction

3D Dynamic Blockworld uses Miniworld as the simulator for partially observed dynamic
environment simulation. Some examples of the dataset:
<p align="center">
  <img src="images/blockworld_vis_1.jpg" width="260" alt="Blockworld Vis 1" style="margin-right: 12px;">
  <img src="images/blockworld_vis_2.jpg" width="260" alt="Blockworld Vis 2">
</p>

## Installation

Install from source:

```console
git clone https://github.com/hlillemark/Miniworld.git
cd Miniworld
conda create -n miniworld python=3.10 -y
conda activate miniworld
python3 -m pip install -e ".[dataset]"
```

If you run into any problems, please take a look at the [troubleshooting guide](docs/content/troubleshooting.md).

## 3D Dynamic Blockworld Quick Usage:

This section describes the dataset generation for the 3D Dynamic Blockworld dataset used in the [Flow Equivariant World Models paper](https://flowequivariantworldmodels.github.io). Static, Dynamic, Textures, and different agent options can be configured. Example commands are available below: 

Command to generate one sample from the textured training set used in the FloWM paper: 
```console 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --out-prefix ./out/tex --debug-join \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball
```

To run parallel dataset generation, set the settings from this command. This will generate the textured validation set used in the FloWM paper:
```console 
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
```

python -m scripts.generate_videos_batch \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --dataset-root /Users/hansen/Desktop/ucsd/Miniworld/out/tex_validation \
  --num-videos 64 --block-size 4 --num-processes 16 \
  -- \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 --no-time-limit \
  --render-width 256 --render-height 256 --obs-width 256 --obs-height 256 \
  --steps 500 --output-2d-map --room-size 16 \
  --block-size-xy 0.7 --block-height 1.5 \
  --agent-box-allow-overlap --box-allow-overlap --grid-cardinal-only \
  --policy biased_walk_v2 --forward-prob 0.90 --cam-fov-y 60 \
  --num-blocks-min 6 --num-blocks-max 10 --ensure-base-palette \
  --randomize-wall-tex --randomize-floor-tex --randomize-box-tex --box-and-ball --debug-join

## 3D Dynamic Blockworld Details:
At each timestep, the agent makes an action and the blocks update according to the velocity they were initialized with. The agent is controlled by the policy and various parameters related to setting options for that policy. 

#### Action convention
For the environment used, the following discrete actions are used and stored in the metadata `<out_prefix>_actions.pt` file. They are stored under `actions` after loading the `.pt` file. The position and viewing direction are also stored in the `_actions.pt` file. 
- turn_left: 0
- turn_right: 1
- move_forward: 2
- move_back: 3 (not used)
- do_nothing: 4 (added, not in the original miniworld env)
- pick_up: 5 (not used)
- drop: 6 (not used)

#### Important configuration tags
Some notable configuration tags for dataset generation are listed below:
- steps: number of environment steps to record
- policy: name of the policy to use
- randomize-wall-tex, randomize-floor-tex, randomize-box-tex: whether or not to use randomized textures for dataset generation
- box-and-ball: whether to spawn both boxes and balls instead of just boxes. 


#### Important parallel generation tags
Some notable configuration options for parallel dataset generation:
- num-videos: number of videos to generate
- block-size: items per block directory
- num-processes: number of parallel processes to spawn

## Original Miniworld Usage

There is a simple UI application which allows you to control the simulation or real robot manually.
The `manual_control.py` application will launch the Gym environment, display camera images and send actions
(keyboard commands) back to the simulator or robot. The `--env-name` argument specifies which environment to load.
See the list of [available environments](docs/environments.md) for more information.

```
./manual_control.py --env-name MiniWorld-Hallway-v0

# Display an overhead view of the environment
./manual_control.py --env-name MiniWorld-Hallway-v0 --top_view
```

There is also a script to run automated tests (`run_tests.py`) and a script to gather performance metrics (`benchmark.py`).

### Offscreen Rendering (Clusters and Colab)

When running MiniWorld on a cluster or in a Colab environment, you need to render to an offscreen display. You can
run `gym-miniworld` offscreen by setting the environment variable `PYOPENGL_PLATFORM` to `egl` before running MiniWorld, e.g.

```
PYOPENGL_PLATFORM=egl python3 your_script.py
```

Alternatively, if this doesn't work, you can also try running MiniWorld with `xvfb`, e.g.

```
xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 your_script.py
```

# Citation

To cite this project please use:

```bibtex
@misc{lillemark2026flowequivariantworldmodels,
    title={Flow Equivariant World Models: Memory for Partially Observed Dynamic Environments}, 
    author={Hansen Jin Lillemark and Benhao Huang and Fangneng Zhan and Yilun Du and Thomas Anderson Keller},
    year={2026},
    eprint={2601.01075},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2601.01075}, 
  }
```
