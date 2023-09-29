# Overcooked Web App 
Adapted from the Human_Aware_Rl / Overcooked AI package from https://github.com/HumanCompatibleAI/human_aware_rl

Make sure you have installed Overcooked by following the instructions in the README.md

#### Train two agents. (**Make sure to be in the base directory of PantheonRL**)
    python trainer.py OvercookedMultiEnv-v0 PPO PPO --env-config '{"layout_name":"simple"}' --seed 10 --preset 1

#### Watch the two trained agents play against each other.
    python overcookedgym/overcooked-flask/app.py --modelpath_p0 models/OvercookedMultiEnv-v0-simple-PPO-ego-10 --modelpath_p1 models/OvercookedMultiEnv-v0-simple-PPO-alt-10 --layout_name simple

#### Play as the alt-agent, with an AI-ego agent. Save the trajectory in file. (When website loads, set Player2 to Human Keyboard Input)
    python overcookedgym/overcooked-flask/app.py --modelpath_p0 models/OvercookedMultiEnv-v0-simple-PPO-ego-10 --layout_name simple --trajs_savepath trajs/OvercookedMultiEnv-v0-simple-PPOHuman-10

#### Watch the replay of the saved trajectory.
    python overcookedgym/overcooked-flask/app.py --layout_name simple --replay_traj trajs/OvercookedMultiEnv-v0-simple-PPOHuman-10
