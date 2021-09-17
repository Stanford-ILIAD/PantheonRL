# the page displayed when the model is being trained
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, session
)
from website.constants import ENV_LIST, LAYOUT_LIST
from website.login import login_required, login_for_training
from website.data_processing import start_training, read, check_trained, gen_tensorboard, stop_tensorboard
from website.db import get_db
from website.constants import generate_url
from urllib.parse import urlparse

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route("/", methods=("GET","POST"))
@login_for_training
def main():
    if not 'tb_log' in session:
        session['tb_log'] = f"./data/user{g.user['id']}logs"
        session['tb_name'] = f"user{g.user['id']}-{g.user['filedata']}"
    if g.user['running']:
        session['updates'], error = read(session['tb_log'], session['tb_name'])

        if error is not None:
            flash(error)

    elif check_trained(session['tb_log'], session['tb_name']):
        return redirect(url_for('training.done'))
    return render_template('training.html', training=g.user['running'], done=False, tb_url=generate_url(urlparse(request.base_url).hostname))

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
            start_training(g.user['id'], session['env_data'], session['ego'], session['partners'], session['tb_log'], session['tb_name'], db)

        return redirect(url_for("training.done"))

@bp.route("/done", methods=("POST", "GET"))
@login_for_training
def done():
    if g.user['running'] == True:
        db = get_db()
        db.execute(
            'UPDATE user SET running = ?'
            ' WHERE id = ?',
            (False, g.user['id'])
        )
        db.commit()
    if "processid" in session:
        stop_tensorboard(session['processid'])
        session['tb'] = False
    return render_template('training.html', training=g.user['running'], done=True)

@bp.route("/tb", methods=("POST",))
@login_for_training
def tb():
    if g.user['running']:
        processid, error = gen_tensorboard(session['tb_log'], session['tb_name'])

        if error is not None:
            flash(error)
        else:
            session['tb'] = True
            session['processid'] = processid

    elif check_trained(session['tb_log'], session['tb_name']):
        return redirect(url_for('training.done'))
    
    return redirect(url_for("training.main"))