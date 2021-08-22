# the page displayed when the model is being trained
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, session
)
from website.constants import ENV_LIST, LAYOUT_LIST
from website.login import login_required
from website.data_processing import start_training, read, check_for_updates
from website.db import get_db

from collections import namedtuple

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route("/", methods=("GET","POST"))
def main():
    if request.method == "POST":
        session['updates'] = read(session['tb_log'], session['tb_name'])
    return render_template('training.html', training=g.user['running'])

@bp.route("/learn", methods=("POST",))
@login_required
def learn():
    if request.method == "POST":
        if g.user['running'] == False:
            db = get_db()
            db.execute(
                'UPDATE user SET running = ?'
                ' WHERE id = ?',
                (True, g.user['id'])
            )
            db.commit()
            start_training(g.user['id'], session['env_data'], session['ego'], session['partners'], session['tb_log'], session['tb_name'])
        return redirect(url_for('training.done'))

@bp.route("/done", methods=("POST", "GET"))
def done():
    if g.user['running'] == True:
        db = get_db()
        db.execute(
            'UPDATE user SET running = ?'
            ' WHERE id = ?',
            (False, g.user['id'])
        )
        db.commit()
    #TODO: change where this redirects to
    return redirect(url_for('welcome.main'))