# mostly taken from the Flask tutorial
import functools

from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
from werkzeug.security import check_password_hash, generate_password_hash

from website.db import get_db

bp = Blueprint('login', __name__, url_prefix='/login')

@bp.route('/', methods=('GET', 'POST'))
def main():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            # first tries to insert the user into the database
            try:
                db.execute(
                    "INSERT INTO user (username, password, running) VALUES (?, ?, ?)",
                    (username, generate_password_hash(password), False),
                )
                db.commit()
            except db.IntegrityError:
                pass

            # whether or not that works, try to get the user from the database
            user = db.execute(
                "SELECT * FROM user WHERE USERNAME = ?", (username,)
            ).fetchone()

            if user is None:
                error = 'Incorrect username.'
            elif not check_password_hash(user['password'], password):
                error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect(url_for('welcome.main'))

        flash(error)

    return render_template('login.html')

@bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login.main'))

@bp.before_app_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            "SELECT * FROM user WHERE id = ?", (user_id,)
        ).fetchone()

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('login.main'))
        elif g.user['running']:
                return redirect(url_for('training.main'))
        session['tb'] = False
        return view(**kwargs)

    return wrapped_view

def login_for_training(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect(url_for('login.main'))

        return view(**kwargs)

    return wrapped_view