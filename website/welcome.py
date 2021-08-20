# the welcome page for the website, where users select their environment
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for
)
from website.constants import ENV_LIST

bp = Blueprint("welcome", __name__)

@bp.route("/", methods=('GET','POST'))
def main():
    if request.method == 'POST':
        env_name = request.form['env']
        error = None

        if not env_name:
            error('You must select an environment to continue.')

        if error is not None:
            flash(error)

        for possible_env in ENV_LIST:
            if env_name == possible_env:
                return redirect(url_for(f"envs.{possible_env}"))
    return render_template('welcome.html', envs=ENV_LIST)