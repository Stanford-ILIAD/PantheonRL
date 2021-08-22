# the agent settings page
from website.login import login_required
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, session
)
from website.constants import EGO_LIST, PARTNER_LIST, ENV_LIST
from website.data_processing import create_ego_dict, create_partner_dict, check_agent_errors
import requests

bp = Blueprint("agents", __name__)

@bp.route("/<string:env>/agents", methods=('GET', 'POST'))
@login_required
def agents(env):
    if not 'env_data' in session:
        return redirect(url_for('welcome.main'))

    if request.method == 'POST':
        error = None
        if not 'ego' in session:
            error = "Please add the ego agent before training."
        elif not 'partners' in session or len(session['partners']) == 0:
            error = "Please add at least one partner agent."
        
        if error is not None:
            flash(error)
        else:
            errors, partners, tb_log, tb_name = check_agent_errors(g.user['id'], env, session['ego'], session['partners'])
        
        if not errors == []:
            errors.append("Either add more agents, or press \'Train\' again to continue.")
            for e in errors:
                flash(e)
            session['partners'] = partners
        else:
            session['tb_log'] = tb_log
            session['tb_name'] = tb_name
            return redirect(url_for('training.main'))
    
    # set blank agent parameters in session
    if not 'egotype' in session:
        session['egotype'] = None
    if not 'partnertype' in session:
        session['partnertype'] = None
    if not 'partners' in session:
        session['partners'] = []
    return render_template('agentparams.html', partners=len(session['partners']), ego_options=EGO_LIST, partner_options=PARTNER_LIST, env=env, env_options=ENV_LIST)

@bp.route("/<string:env>/egotype", methods=('POST',))
@login_required
def egotype(env):
    if request.method == 'POST':
        session['egotype'] = request.form['egotype']
        return redirect(url_for('agents.agents', env=env))

@bp.route("/<string:env>/partnertype", methods=('POST',))
@login_required
def partnertype(env):
    if request.method == 'POST':
        session['partnertype'] = request.form['partnertype']
        if session['partnertype'] == 'FIXED':
            error = "Fixed partner agents are not yet implemented. Please pick a different partner type."
            flash(error)
        return redirect(url_for('agents.agents', env=env))

@bp.route("/<string:env>/setego", methods=('POST',))
@login_required
def setego(env):
    if request.method == 'POST':
        # add ego params to the session variable
        session['ego'] = create_ego_dict(session['egotype'], request.form)
        session['egotype'] = None
        return redirect(url_for('agents.agents', env=env))


@bp.route("/<string:env>/setpartner", methods=('POST',))
@login_required
def setpartner(env):
    if request.method == 'POST':
        # add ego params to the session variable
        error, partner_dict = create_partner_dict(session['partnertype'], env, request.form)
        if error is not None:
            flash(error)
        else:
            session['partners'].append(partner_dict)
            session['partnertype'] = None
        return redirect(url_for('agents.agents', env=env))