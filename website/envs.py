# the environment settings page
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, session
)
# from multiagentworld import blockworld, simpleblockworld, rps, liar, overcooked
import gym

bp = Blueprint("envs", __name__)

@bp.route("/blockworld", methods=('GET', 'POST'))
def blockworld():
    if request.method == 'POST':
        # session['environment'] = gym.make('multiagentworld:BlockEnv-v1')
        return redirect(url_for('agents', env='blockworld'))
    return render_template('environments/blockworld.html')

@bp.route("/simpleblockworld", methods=('GET', 'POST'))
def simpleblockworld():
    if request.method == 'POST':
        # session['environment'] = gym.make('multiagentworld:BlockEnv-v0')
        return redirect(url_for('agents', env='simpleblockworld'))
    return render_template('environments/simpleblockworld.html')

@bp.route("/overcooked", methods=('GET', 'POST'))
def overcooked():
    if request.method == 'POST':
        layout_name = request.form['layout_name'].strip()
        ego_agent_idx = int(request.form['ego_agent_idx'])
        baselines = request.form['baselines'] == 'on'
        error = None

        if not layout_name or not layout_name.isalnum():
            error = "Please enter a valid layout name."

        if error is not None:
            flash(error)

        # session['environment'] = gym.make('OvercookedMultiEnv-v0', layout_name=layout_name,
        # ego_agent_idx=ego_agent_idx, baselines=baselines)

        return redirect(url_for('agents', env='overcooked'))
    return render_template('environments/overcooked.html')

@bp.route("/liar", methods=('GET', 'POST'))
def liar():
    if request.method == 'POST':
        probegostart = request.form['probegostart']

        # session['environment'] = gym.make('LiarsDice-v0', probegostart=probegostart)

        return redirect(url_for('agents', env='liar'))
    return render_template('environments/liar.html')

@bp.route("/rps", methods=('GET', 'POST'))
def rps():
    if request.method == 'POST':
        # session['environment'] = gym.make('RPS-v0')

        return redirect(url_for('agents', env='rps'))
    return render_template('environments/rps.html')




