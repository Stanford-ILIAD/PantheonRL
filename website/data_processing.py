# functions for saving data, changing formats, etc
# will change as i change the data formats being used
import numpy as np
import gym
import os
from collections import namedtuple
from trainer import generate_env, generate_ego, gen_partner
import datetime
import time
import tensorflow as tf
from os import listdir
from os.path import isfile, join

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

def create_args_object(env_args):
    Config = namedtuple("Config", ["env", "env_config", "record", "framestack"])
    return Config(env_args["env"], env_args["env_config"], env_args["record"], env_args["framestack"])

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

    ego_dict = {"type": ego_type, "seed": seed, "timesteps": int(args["timesteps"])}

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

def check_agent_errors(id, env, ego, partners):
    # assumes that ego exists and partners has length at least one
    errors = []
    i = 0
    while i < len(partners):
        if partners[i]["type"] == "FIXED":
            errors.append(f"Fixed agents haven't been implemented yet. Partner {i + len(errors)} will be deleted.\n")
            partners.pop(i)
            i -= 1
        i += 1

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log = f"./data/user{id}logs"
    tensorboard_name = f"user{id}-{time}"
    
    return errors, partners, tensorboard_log, tensorboard_name

def create_ego_object(ego_data, num_partners, tensorboard_log):
    ego_config = {"verbose": 1}
    alt = [0]*num_partners
    device = "auto"
    Ego = namedtuple("Ego", ["ego_config", "tensorboard_log", "alt", "device", "seed", "ego"])
    return Ego(ego_config, tensorboard_log, alt, device, ego_data['seed'], ego_data['type'])

def create_partner_object(seed):
    Partner = namedtuple("Partner", ["seed", "device", "share_latent"])
    return Partner(seed, "auto", False)

def start_training(id, env_data, ego_data, partners, tensorboard_log, tensorboard_name):
    print("started training")
    env_args = create_args_object(env_data)
    env, alt_env = generate_env(env_args)
    print(f"Environment: {env}; Partner env: {alt_env}")

    ego_agent = generate_ego(env, create_ego_object(ego_data, len(partners), tensorboard_log))
    print(f'Ego: {ego_agent}')

    for partner in partners:
        type = partner.pop("type")
        seed = None
        if "seed" in partner:
            seed = partner.pop("seed")
        p_args = create_partner_object(seed)
        env.add_partner_agent(gen_partner(type, partner, alt_env, ego_agent, p_args))
    
    learn_config = {'total_timesteps': ego_data["timesteps"]}
    if tensorboard_log is not None:
        learn_config['tb_log_name'] = tensorboard_name
    ego_agent.learn(**learn_config)

    if env_data["record"] is not None:
        transition = env.get_transitions()
        transition.write_transition(env_data["record"])

def read(tensorboard_log, tensorboard_name):
    mypath = f"{tensorboard_log}/{tensorboard_name}_1"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    print(f"reading from {join(mypath, onlyfiles[0])}")
    summaries = tf.compat.v1.train.summary_iterator(join(mypath, onlyfiles[0]))
    mydict = {}
    for e in summaries:
        for v in e.summary.value:
            mydict[v.tag] = v.simple_value
    return mydict

def check_for_updates(file):
    # a function that would theoretically check for updates from the learning agent
    # right now it just returns a test json
    return {"updates": "In terms of updates, there are no updates."}    
        