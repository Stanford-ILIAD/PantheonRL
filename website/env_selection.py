# the welcome page for the website, where users select their environment
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, session
)
from website.constants import ENV_LIST, LAYOUT_LIST
from website.login import login_required
from website.data_processing import common_env_configs, create_args_dict

from collections import namedtuple

bp = Blueprint("welcome", __name__)

@bp.route("/", methods=('GET','POST'))
@login_required
def main():
    clear_agent_selection()
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
            session['env_data'] = create_args_dict(env_name, env_config, record, framestack)
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
            session['env_data'] = create_args_dict(env_name, env_config, record, framestack)
            return redirect(url_for('agents.agents', env='simpleblockworld'))
    return render_template('environments/simpleblockworld.html', envs=ENV_LIST, selected=True)

@bp.route("/overcooked", methods=('GET', 'POST'))
@login_required
def overcooked():
    if request.method == 'POST':
        layout_name = request.form['layout_name']
        ego_agent_idx = int(request.form['ego_agent_idx'])
        baselines = 'baselines' in request.form

        record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "OvercookedMultiEnv-v0"
            env_config = {"layout_name": layout_name, "ego_agent_idx": ego_agent_idx, "baselines": baselines}
            session['env_data'] = create_args_dict(env_name, env_config, record, framestack)

            return redirect(url_for('agents.agents', env='overcooked'))

    return render_template('environments/overcooked.html', envs=ENV_LIST, selected=True, layouts=LAYOUT_LIST)

@bp.route("/liar", methods=('GET', 'POST'))
def liar():
    if request.method == 'POST':
        probegostart = request.form['probegostart']

        record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "LiarsDice-v0"
            env_config = {"probegostart": probegostart}
            session['env_data'] = create_args_dict(env_name, env_config, record, framestack)

            return redirect(url_for('agents.agents', env='liar'))

    return render_template('environments/liar.html', envs=ENV_LIST, selected=True)

@bp.route("/rps", methods=('GET', 'POST'))
def rps():
    if request.method == 'POST':
        record, framestack, error = common_env_configs(request.form, g.user['id'])

        if error is not None:
            flash(error)
        else:
            env_name = "RPS-v0"
            env_config = {}
            session['env_data'] = create_args_dict(env_name, env_config, record, framestack)

            return redirect(url_for('agents.agents', env='rps'))
    return render_template('environments/rps.html', envs=ENV_LIST, selected=True)

def clear_agent_selection():
    delete_from_session(['egotype', 'partnertype', 'partners', 'ego', 'tb_log', 'tb_name', 'updates', 'processid'])

def delete_from_session(vars):
    for var in vars:
        if var in session:
            session.pop(var)
    
