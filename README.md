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

## Introduction

MiniWorld is a minimalistic 3D interior environment simulator for reinforcement
learning &amp; robotics research. It can be used to simulate environments with
rooms, doors, hallways and various objects (eg: office and home environments, mazes).
MiniWorld can be seen as a simpler alternative to VizDoom or DMLab. It is written
100% in Python and designed to be easily modified or extended by students.

<!-- TODO: put here side by side of three textures for our blockworld env. -->
<p align="center">
    <img src="images/maze_top_view.jpg" width=260 alt="Figure of Maze environment from top view">
    <img src="images/sidewalk_0.jpg" width=260 alt="Figure of Sidewalk environment">
    <img src="images/collecthealth_0.jpg" width=260 alt="Figure of Collect Health environment">
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

## 3D Dynamic Blockworld Usage

This section describes the dataset generation for the 3D Dynamic Blockworld dataset used in the [Flow Equivariant World Models paper](https://flowequivariantworldmodels.github.io). Static, Dynamic, and different agent options can be configured. Example commands are available below: 

To generate one example video of blockworld, you can use the following command
```console 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.90 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 300 --out-prefix ./out/run_move --debug --room-size 16
```

To run parallel dataset generation, set the settings from this command:
```console 
python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 16 \
  --dataset-root ./out/blockworld_dataset --num-videos 80 --block-size 10 --num-processes 8

xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python -m scripts.generate_videos \
  --env-name MiniWorld-MovingBlockWorld-v0 \
  --policy biased_random --forward-prob 0.9 --wall-buffer 0.5 --avoid-turning-into-walls --agent-box-allow-overlap \
  --turn-step-deg 90 --forward-step 1.0 --heading-zero \
  --grid-mode --grid-vel-min -1 --grid-vel-max 1 \
  --render-width 128 --render-height 128 --obs-width 128 --obs-height 128 \
  --steps 500 --room-size 16 \
  --dataset-root ./out/blockworld_dataset --num-videos 80 --block-size 10 --num-processes 8
```

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
@article{MinigridMiniworld23,
  author       = {Maxime Chevalier-Boisvert and Bolun Dai and Mark Towers and Rodrigo de Lazcano and Lucas Willems and Salem Lahlou and Suman Pal and Pablo Samuel Castro and Jordan Terry},
  title        = {Minigrid \& Miniworld: Modular \& Customizable Reinforcement Learning Environments for Goal-Oriented Tasks},
  journal      = {CoRR},
  volume       = {abs/2306.13831},
  year         = {2023},
}
```
