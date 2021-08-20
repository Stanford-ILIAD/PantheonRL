# the agent settings page
from flask import (
    Blueprint, flash, g, redirect, render_template, request, url_for, session
)
# from multiagentworld import blockworld, simpleblockworld, rps, liar, overcooked
import gym

bp = Blueprint("agents", __name__)

@bp.route("/<string:env>/agents", methods=('GET', 'POST'))
def agents(env):
    return render_template('baseagents.html')