# the welcome page for the website, where users select their environment
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify
)
from website.constants import ENV_LIST, LAYOUT_LIST
from website.login import login_required

import numpy as np
import gym
import os
from multiagentworld.envs.rpsgym.rps import RPSEnv, RPSWeightedAgent
from multiagentworld.envs.blockworldgym import simpleblockworld, blockworld
from multiagentworld.envs.liargym.liar import LiarEnv, LiarDefaultAgent
from trainer import generate_env

bp = Blueprint("welcome", __name__)

@bp.route("/", methods=('GET','POST'))
@login_required
def main():
    if request.method == 'POST':
        env_name = request.form['env']
        error = None

        if not env_name:
            error('You must select an environment to continue.')

        if error is not None:
            flash(error)

        for possible_env in ENV_LIST:
            if env_name == ENV_LIST[possible_env]:
                return redirect(url_for(f"welcome.{ENV_LIST[possible_env]}"))
    return render_template('welcome.html', envs=ENV_LIST, selected=False)

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

def create_args_dict(env_name, env_config, record, framestack):
    return {"env": env_name, "env_config": env_config, "record": record, "framestack": framestack}

@bp.route("/blockworld", methods=('GET', 'POST'))
@login_required
def blockworld():
    if request.method == 'POST':
        record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "BlockEnv-v1"
            env_config = {}
            env, altenv = generate_env(create_args_dict(env_name, env_config, record, framestack))
            return redirect(url_for('agents.agents', env='blockworld'))
    return render_template('environments/blockworld.html', envs=ENV_LIST, selected=True)

@bp.route("/simpleblockworld", methods=('GET', 'POST'))
@login_required
def simpleblockworld():
    if request.method == 'POST':
        record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "BlockEnv-v0"
            env_config = {}
            env, altenv = generate_env(create_args_dict(env_name, env_config, record, framestack))
            return redirect(url_for('agents.agents', env='simpleblockworld'))
    return render_template('environments/simpleblockworld.html', envs=ENV_LIST, selected=True)

@bp.route("/overcooked", methods=('GET', 'POST'))
@login_required
def overcooked():
    if request.method == 'POST':
        layout_name = request.form['layout_name'].strip()
        ego_agent_idx = int(request.form['ego_agent_idx'])
        baselines = request.form['baselines'] == 'on'
        error = None

        if not layout_name or not layout_name.isalnum():
            error = "Please enter a valid layout name."
        else:
            record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "OvercookedMultiEnv-v0"

        return redirect(url_for('agents.agents', env='overcooked'))
    return render_template('environments/overcooked.html', envs=ENV_LIST, selected=True, layouts=LAYOUT_LIST)

@bp.route("/liar", methods=('GET', 'POST'))
def liar():
    if request.method == 'POST':
        probegostart = request.form['probegostart']

        # create and save the environment

        return redirect(url_for('agents.agents', env='liar'))
    return render_template('environments/liar.html', envs=ENV_LIST, selected=True)

@bp.route("/rps", methods=('GET', 'POST'))
def rps():
    if request.method == 'POST':
        # create and save the environment

        return redirect(url_for('agents.agents', env='rps'))
    return render_template('environments/rps.html', envs=ENV_LIST, selected=True)

def saveenvs(env, alt_env, id):
    if not os.path.exists('data'):
        os.makedirs('data')

    np.savez(f"./data/user{id}.npz", env=env, alt_env=alt_env)