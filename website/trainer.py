# the page displayed when the model is being trained
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, jsonify, session
)
from website.constants import ENV_LIST, LAYOUT_LIST
from website.login import login_required
from website.data_processing import start_training
from website.db import get_db

from collections import namedtuple

bp = Blueprint("training", __name__, url_prefix="/training")

@bp.route("/", methods=("GET", "POST"))
@login_required
def main():
    if g.user['running'] == False:
        db = get_db()
        db.execute(
            'UPDATE user SET running = ?'
            ' WHERE id = ?',
            (True, g.user['id'])
        )
        db.commit()
    return render_template('training.html')
