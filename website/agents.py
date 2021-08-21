# the agent settings page
from website.login import login_required
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, session
)
from website.constants import EGO_LIST, PARTNER_LIST

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
        
        # else generate the agent + partners and run
    
    # set blank agent parameters in session
    if not 'egotype' in session:
        session['egotype'] = None
    if not 'partnertype' in session:
        session['partnertype'] = None
    if not 'partners' in session:
        session['partners'] = []
    return render_template('baseagents.html', partners=len(session['partners']), ego_options=EGO_LIST, partner_options=PARTNER_LIST, env=env)

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
        return redirect(url_for('agents.agents', env=env))