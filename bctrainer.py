import argparse
import json

from pantheonrl.algos.bc import BC
from pantheonrl.common import trajsaver
from pantheonrl.common.multiagentenv import SimultaneousEnv

from trainer import (generate_env, ENV_LIST, LAYOUT_LIST)


class EnvException(Exception):
    """ Raise when parameters do not align with environment """


def input_check(args):
    # Env checking
    if args.env == 'OvercookedMultiEnv-v0':
        if 'layout_name' not in args.env_config:
            raise EnvException(f"layout_name needed for {args.env}")
        elif args.env_config['layout_name'] not in LAYOUT_LIST:
            raise EnvException(
                f"{args.env_config['layout_name']} is not a valid layout")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            BC algorithm given a trajectory
            ''')

    parser.add_argument('env',
                        choices=ENV_LIST,
                        help='The environment to train in')

    parser.add_argument('trajectory',
                        type=str,
                        help='Location of trajectory')

    parser.add_argument('--choose-alt',
                        action='store_true',
                        help='Train from the alt trajectory (default is ego)')

    parser.add_argument('--total-epochs', '-t',
                        type=int,
                        default=10,
                        help='Number of episodes to run')

    parser.add_argument('--l2',
                        type=float,
                        default=0,
                        help='Value of l2 weight of BC algorithm')

    parser.add_argument('--device', '-d',
                        default='auto',
                        help='Device to run pytorch on')

    parser.add_argument('--env-config',
                        type=json.loads,
                        default={},
                        help='Config for the environment')

    parser.add_argument('--framestack', '-f',
                        type=int,
                        default=1,
                        help='Number of observations to stack')

    parser.add_argument('--save',
                        help='File to save the agent into')

    args = parser.parse_args()
    args.record = None

    input_check(args)

    print(f"Arguments: {args}")
    env, altenv = generate_env(args)
    print(f"Environment: {env}; Partner env: {altenv}")

    if isinstance(env, SimultaneousEnv):
        TransitionsClass = trajsaver.SimultaneousTransitions
    else:
        TransitionsClass = trajsaver.TurnBasedTransitions

    if args.choose_alt:
        env = altenv

    transition = TransitionsClass.read_transition(
            args.trajectory, env.observation_space, env.action_space)

    if args.choose_alt:
        data = transition.get_alt_transitions()
    else:
        data = transition.get_ego_transitions()

    clone = BC(observation_space=env.observation_space,
               action_space=env.action_space,
               expert_data=data,
               l2_weight=args.l2,
               device=args.device)

    clone.train(n_epochs=args.total_epochs)
    if args.save is not None:
        clone.save_policy(args.save)
