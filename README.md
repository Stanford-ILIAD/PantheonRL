# PantheonRL

PantheonRL is a package for training and testing multi-agent reinforcement learning environments. The goal of PantheonRL is to provide a modular and extensible framework for training agent policies, fine-tuning agent policies, ad-hoc pairing of agents, and more. PantheonRL also provides a web user interface suitable for lightweight experimentation and prototyping.


PantheonRL is built on top of StableBaselines3 (SB3), allowing direct access to many of SB3's standard RL training algorithms such as PPO. PantheonRL currently follows a decentralized training paradigm -- each agent is equipped with its own replay buffer and update algorithm. The agents objects are designed to be easily manipulable. They can be saved, loaded and plugged into different training procedures such as self-play, ad-hoc / cross-play, round-robin training, or finetuning.


## Installation
```
pip install -e .
```


## Web User Interface

The first time the web interface is being run in a new location, the database must be initialized. After that, the ``init-db`` command should not be called again, because this will clear all user account data.

Set environment variables and (re)inititalize the database
```
export FLASK_APP=website
export FLASK_ENV=development
flask init-db
```

Start the web user interface
```
flask run --host=0.0.0.0
```


## Command Line Invocation


#### Example
```
python3 trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 10 --preset 1
```

#### Testing the Installation
```
python3 tester.py ...
```


## Features

| **General Features**        | **PantheonRL** |
| --------------------------- | ----------------------|
| Documentation               | :heavy_check_mark: |
| Web user interface          | :heavy_check_mark: |
| Built on top of SB3         | :heavy_check_mark: |



| **Environment Features**    | **PantheonRL** |
| --------------------------- | ----------------------|
| Frame stacking (recurrence) | :heavy_check_mark: |
| Simultaneous multiagent envs| :heavy_check_mark: |
| Turn-based multiagent envs  | :heavy_check_mark: |
| 2-player envs               | :heavy_check_mark: |
| N-player envs               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |


| **Training Features**           | **PantheonRL** |
| ------------------------------- | ----------------------|
| Self-play                       | :heavy_check_mark: |
| Ad-hoc / cross-play             | :heavy_check_mark: |
| Round-robin training            | :heavy_check_mark: |
| Finetune / adapt to new partners| :heavy_check_mark: |
| Custom policies                 | :heavy_check_mark: |



#### Current Environments

| **Name**              | **Environment Type**  | **Reward Type**  | **Players**     | **Visualization**   |
| --------------------- | --------------------- | ---------------- | --------------- | ------------------- |
| Rock Paper Scissors   | SimultaneousEnv       | Competitive      | 2               | :x:                 |
| Liar's Dice           | TurnBasedEnv          | Competitive       | 2               | :x:                 |
| Block World [[1]](#1) | TurnBasedEnv          | Cooperative      | 2               | :heavy_check_mark:  |
| Overcooked [[2]](#2)  | SimultaneousEnv       | Cooperative      | 2               | :heavy_check_mark:  |

<a id="1">[1]</a>
Adapted from the block construction task from https://github.com/cogtoolslab/compositional-abstractions

<a id="2">[2]</a>
Adapted from the Human-Aware_Rl / Overcooked AI package from https://github.com/HumanCompatibleAI/human_aware_rl


## Acknowledgments
