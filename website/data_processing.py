# functions for saving data, changing formats, etc
# will change as i change the data formats being used
import numpy as np
import copy
import os
from collections import namedtuple
from trainer import generate_env, generate_ego, gen_partner, gen_fixed
import datetime
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from website.constants import PARTNER_LIST, ENV_TO_NAME, TB_PORT, ADAP_TYPES
import subprocess
import signal

def savedpartnerpath(id, env, name, ptype):
    return f"./data/user{id}/{env}/fixedpartners/{ptype}/{name}"

def loadpartnerpath(id, env):
    return f"./data/user{id}/{env}/fixedpartners"

def savedegopath(id, env, name, ptype):
    return f"./data/user{id}/{env}/fixedego/{ptype}/{name}"

def fixedpartneroptions(id, env):
    mypath = loadpartnerpath(id, env)
    if not os.path.isdir(mypath):
        return "You have not saved any fixed partners for this environment. Please select a different partner type.", []
    else:
        options = []
        for ptype in PARTNER_LIST:
            # TODO: add a better fix later
            if not ptype in ADAP_TYPES:
                if os.path.isdir(join(mypath, ptype)):
                    currpath = join(mypath, ptype)
                    options += [(f.replace('.zip', ''), ptype) for f in listdir(currpath) if isfile(join(currpath, f))]
        return None, options

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

def create_ego_dict(ego_type, args, env, id):
    seed = None
    error = None
    location = None
    if "seed" in args and args['seed'] != "":
        seed = int(args["seed"])
    if "egoname" in args and args['egoname'] != "":
        location = savedegopath(id, ENV_TO_NAME[env], args['egoname'], ego_type)
        if not args['egoname'].isalnum():
            error = "Name to save as must be alphanumeric."
        elif os.path.exists(f"{location}.zip"):
            error = "There already exists an ego agent saved with that name."

    ego_dict = {"type": ego_type, "seed": seed, "timesteps": int(args["timesteps"]), "location": location}

    return error, ego_dict

def create_partner_dict(id, partner_type, env, args):
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
        mypath = loadpartnerpath(id, ENV_TO_NAME[env])
        if not os.path.isdir(mypath):
            error = "You do not have any fixed partners for this environment. Please select a different partner type."
        else:
            partner_dict['location'] = join(mypath, args['fixedtype'])
            partner_dict['ptype'] = args['fixedtype'][:args['fixedtype'].find('/')]
    elif partner_type != "DEFAULT":
        seed = None
        if "seed" in args and args['seed'] != "":
            seed = int(args["seed"])
        partner_dict['seed'] = seed
        save = None
        if "partnername" in args and args['partnername'] != "":
            if not args['partnername'].isalnum():
                error = "Name to save as must be alphanumeric."
            elif os.path.exists(f"{savedpartnerpath(id, env, args['partnername'], partner_type)}.zip"):
                error = "There already exists a partner saved with that name."
            save = args['partnername']
        partner_dict['save'] = save
    
    return error, partner_dict

def check_agent_errors(id, env, ego, partners):
    errors = []
    if ego is None or len(partners) < 1:
        errors.append("Need at least one valid ego agent and partner agent to train with.")

    i = 0
    while i < len(partners):
        if partners[i]["type"] == "FIXED" and (not 'location' in partners[i] or not 'ptype' in partners[i]):
            errors.append(f"Fixed agents need a filename to load from. Partner {i + len(errors)} will be deleted.\n")
            partners.pop(i)
            i -= 1
        elif partners[i]["type"] == "FIXED" and not os.path.isfile(partners[i]['location']):
            # honestly if it reaches here it's probably my fault and not the users
            errors.append(f"Fixed agents need a valid filepath. Partner {i + len(errors)} will be deleted.\n")
            partners.pop(i)
            i -= 1
        i += 1

    time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_log = f"./data/user{id}logs"
    tensorboard_name = f"user{id}-{time}"
    
    return errors, partners, tensorboard_log, tensorboard_name, time

def create_ego_object(ego_data, num_partners, tensorboard_log):
    ego_config = {"verbose": 1}
    alt = [0]*num_partners
    device = "auto"
    Ego = namedtuple("Ego", ["ego_config", "tensorboard_log", "alt", "device", "seed", "ego"])
    return Ego(ego_config, tensorboard_log, alt, device, ego_data['seed'], ego_data['type'])

def create_partner_object(seed, tensorboard_log, tensorboard_name, partner_num):
    Partner = namedtuple("Partner", ["seed", "device", "share_latent", "tensorboard_log", "tensorboard_name", "partner_num", "verbose_partner"])
    return Partner(seed, "auto", False, tensorboard_log, tensorboard_name, partner_num, False)

def start_training(id, env_data, ego_data, partners, tensorboard_log, tensorboard_name, mydatabase):
    ego_save = ego_data.pop("location")
    env_args = create_args_object(env_data)
    env, alt_env = generate_env(env_args)

    ego_agent = generate_ego(env, create_ego_object(ego_data, len(partners), tensorboard_log))

    partners_to_save = []
    for i, partner in enumerate(partners):
        ptype = partner.pop("type")
        seed = None
        if "seed" in partner:
            seed = partner.pop("seed")
        save = None
        if "save" in partner:
            save = partner.pop("save")
        #TODO: adap agents can't load, fix this later
        if ptype == "FIXED":
            current_partner = gen_fixed({}, partner.pop('ptype'), partner.pop('location'))
        else:
            p_args = create_partner_object(seed, tensorboard_log, tensorboard_name, i)
            current_partner = gen_partner(ptype, copy.deepcopy(partner), alt_env, ego_agent, p_args)
        env.add_partner_agent(current_partner)
        if save is not None:
            partners_to_save.append((current_partner, save, ptype))
    
    learn_config = {'total_timesteps': ego_data["timesteps"]}
    if tensorboard_log is not None:
        learn_config['tb_log_name'] = tensorboard_name
    ego_agent.learn(**learn_config)

    if env_data["record"] is not None:
        transition = env.get_transitions()
        transition.write_transition(env_data["record"])
    
    for partner, name, ptype in partners_to_save:
        partner.model.save(savedpartnerpath(id, env_data['env'], name, ptype))
    
    if ego_save is not None:
        ego_agent.save(ego_save)

    mydatabase.execute(
            'UPDATE user SET running = ?'
            ' WHERE id = ?',
            (False, id)
        )
    mydatabase.commit()

def check_trained(log, name):
    mypath = f"{log}/{name}_1"
    return os.path.isdir(mypath)

def create_dir(tensorboard_log, tensorboard_name):
    mypath = f"{tensorboard_log}/{tensorboard_name}_1"
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    return mypath, onlyfiles

def read(tensorboard_log, tensorboard_name):
    mypath = f"{tensorboard_log}/{tensorboard_name}_1"
    if not os.path.isdir(mypath):
        return {}, "Please start training before checking for updates."
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    summaries = tf.compat.v1.train.summary_iterator(join(mypath, onlyfiles[len(onlyfiles) - 1]))
    mydict = {}
    for e in summaries:
        for v in e.summary.value:
            mydict[v.tag] = v.simple_value
    return mydict, None

def check_for_updates(file):
    # a function that would theoretically check for updates from the learning agent
    # right now it just returns a test json
    return {"updates": "In terms of updates, we have no updates."}    

def gen_tensorboard(tb_log, tb_name):
    mypath = f"{tb_log}/{tb_name}_1"
    if not os.path.isdir(mypath):
        return None, "Please start training before generating the tensorboard."
    p = subprocess.Popen(["tensorboard", "--logdir", mypath, "--bind_all", "--port", TB_PORT], stderr=subprocess.PIPE)
    try:
        outs, errs = p.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        print("timeout expired")
        return p.pid, None
    return None, f"Tensorboard could not bind to port {TB_PORT} as it was already in use. Tensorboard not created."

def stop_tensorboard(processid):
    try:
        os.kill(processid, signal.SIGINT)
    except OSError:
        pass
        