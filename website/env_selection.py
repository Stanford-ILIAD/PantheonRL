# the welcome page for the website, where users select their environment
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from website.constants import ENV_LIST
from website.login import login_required

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

@bp.route("/blockworld", methods=('GET', 'POST'))
def blockworld():
    if request.method == 'POST':
        # create and save the environment, use a session id?
        return redirect(url_for('agents.agents', env='blockworld'))
    return render_template('environments/blockworld.html', envs=ENV_LIST, selected=True)

@bp.route("/simpleblockworld", methods=('GET', 'POST'))
def simpleblockworld():
    if request.method == 'POST':
        # create and save the environment
        return redirect(url_for('agents.agents', env='simpleblockworld'))
    return render_template('environments/simpleblockworld.html', envs=ENV_LIST, selected=True)

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

        # create and save the environment

        return redirect(url_for('agents.agents', env='overcooked'))
    return render_template('environments/overcooked.html', envs=ENV_LIST, selected=True)

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