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

def create_ego_dict(ego_type, args):
    seed = None
    error = None
    if "seed" in args and args['seed'] != "":
        seed = int(args["seed"])

    ego_dict = {"type": ego_type, "seed": seed}

    return ego_dict

def create_partner_dict(partner_type, env, args):
    error = None
    partner_dict = {"type": partner_type}

    if env == "rps" and partner_type == "DEFAULT":
        if not ('r' in args and 'p' in args and 's' in args):
            error = "You must enter the probabilities of rock, paper, and scissors for the RPS Default Agent."
        if not (args['r'].isnumeric() and args['s'].isnumeric() and args['p'].isnumeric()):
            error = "Please enter valid numbers for each probability."
        r, p, s = int(args['r']), int(args['p']), int(args['s'])
        if not (r >= 0 and s >= 0 and p >= 0):
            error = "All probabilities must be nonnegative."
        partner_dict['r'], partner_dict['p'], partner_dict['s'] = r, p, s
    elif partner_type == "FIXED":
        error = "Fixed partners have not yet been implemented - please select a different partner type."
    elif partner_type != "DEFAULT":
        seed = None
        if "seed" in args and args['seed'] != "":
            seed = int(args["seed"])
        partner_dict['seed'] = seed
    
    return error, partner_dict

def check_agent_errors(env, ego, partners):
    # assumes that ego exists and partners has length at least one
    errors = ""
    i = 0
    while i < len(partners):
        if partners[i]["type"] == "DEFAULT":
            errors += f"Default agents haven't been implemented yet. Partner {i} will be deleted.\n"
            partners.pop(i)
            i -= 1
        i += 1
    return errors, partners
        