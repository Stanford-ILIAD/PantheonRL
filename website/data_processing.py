# functions for saving data, changing formats, etc
# will change as i change the data formats being used
import numpy as np
import gym
import os
from collections import namedtuple

def common_env_configs(args, id):
    record = None
    if "record" in args:
        record = f"./data/user{id}traj.json"
    error = None
    framestack = args["framestack"]
    if not framestack or not framestack.isnumeric():
        error = "Please enter a valid integer for the number of frames stacked."
    else:
        framestack = int(framestack)
    
    return record, framestack, error

def create_args_object(env_name, env_config, record, framestack):
    Config = namedtuple("Config", ["env", "env_config", "record", "framestack"])
    args = Config(env_name, env_config, record, framestack)
    return args

def create_args_dict(env_name, env_config, record, framestack):
    args = {"env": env_name, "env_config": env_config, "record": record, "framestack": framestack}
    return args

def saveenvs(env, alt_env, id):
    if not os.path.exists('data'):
        os.makedirs('data')

    np.savez(f"./data/user{id}.npz", env=env, alt_env=alt_env)