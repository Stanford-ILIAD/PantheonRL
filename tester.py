import argparse
import json
from time import sleep

import numpy as np

from trainer import (generate_env, gen_fixed, gen_default, EnvException,
                     ENV_LIST, ADAP_TYPES, LAYOUT_LIST)

EGO_LIST = ['PPO', 'ModularAlgorithm', 'BC'] + ADAP_TYPES
PARTNER_LIST = ['PPO', 'DEFAULT', 'BC'] + ADAP_TYPES


def input_check(args):
    # Env checking
    if args.env == 'OvercookedMultiEnv-v0':
        if 'layout_name' not in args.env_config:
            raise EnvException(f"layout_name needed for {args.env}")
        elif args.env_config['layout_name'] not in LAYOUT_LIST:
            raise EnvException(
                f"{args.env_config['layout_name']} is not a valid layout")

    # Construct ego config
    if 'verbose' not in args.ego_config:
        args.ego_config['verbose'] = 1

    if args.ego_load is None:
        raise EnvException("Need to provide file for ego to load")

    if (args.alt_load is None) != (args.alt == 'DEFAULT'):
        raise EnvException("Load policy if and only if alt is not DEFAULT")


def generate_agent(env, policy_type, config, location):
    if policy_type == 'DEFAULT':
        return gen_default(config, env)

    return gen_fixed(config, policy_type, location)


def run_test(ego, env, num_episodes, render=False):
    rewards = []
    for game in range(num_episodes):
        obs = env.reset()
        done = False
        reward = 0
        if render:
            env.render()
        while not done:
            action = ego.get_action(obs, False)
            obs, newreward, done, _ = env.step(action)
            reward += newreward

            if render:
                env.render()
                sleep(1/60)

        rewards.append(reward)

    env.close()
    print(f"Average Reward: {sum(rewards)/num_episodes}")
    print(f"Standard Deviation: {np.std(rewards)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Test ego and partner in an environment.

            Environments:
            -------------
            All MultiAgentEnv environments are supported. Some have additional
            parameters that can be passed into --env-config. Specifically,
            OvercookedMultiEnv-v0 has a required layout_name parameter, so
            one must add:

                --env-config '{"layout_name":"[SELECTED_LAYOUT]"}'

            OvercookedMultiEnv-v0 also has parameters `ego_agent_idx` and
            `baselines`, but these have default initializations. LiarsDice-v0
            has an optional parameter, `probegostart`.

            The environment can be wrapped with a framestack, which transforms
            the observation to stack previous observations as a workaround
            for recurrent networks not being supported. It can also be wrapped
            with a recorder wrapper, which will write the transitions to the
            given file.

            Ego-Agent:
            ----------
            The ego-agent is considered the main agent in the environment.
            From the perspective of the ego agent, the environment functions
            like a regular gym environment.

            Supported ego-agent algorithms include PPO, ModularAlgorithm, ADAP,
            and ADAP_MULT. The default parameters of these algorithms can
            be overriden using --ego-config.

            Alt-Agent:
            -----------
            The alt-agents are the partner agents that are embedded in the
            environment. If multiple are listed, the environment randomly
            samples one of them to be the partner at the start of each episode.

            Supported alt-agent algorithms include PPO, ADAP, ADAP_MULT,
            and DEFAULT. DEFAULT refers to the default hand-made policy
            in the environment (if it exists).

            Default parameters for these algorithms can be overriden using
            --alt-config.

            NOTE:
            All configs are based on the json format, and will be interpreted
            as dictionaries for the kwargs of their initializers.

            Example usage (Overcooked with ADAP agents that share the latent
            space):

            python3 tester.py OvercookedMultiEnv-v0 ADAP ADAP --env-config
            '{"layout_name":"random0"}' -l
            ''')

    parser.add_argument('env',
                        choices=ENV_LIST,
                        help='The environment to train in')

    parser.add_argument('ego',
                        choices=EGO_LIST,
                        help='Algorithm for the ego agent')

    parser.add_argument('alt',
                        choices=PARTNER_LIST,
                        help='Algorithm for the partner agent')

    parser.add_argument('--total-episodes', '-t',
                        type=int,
                        default=100,
                        help='Number of episodes to run')

    parser.add_argument('--device', '-d',
                        default='auto',
                        help='Device to run pytorch on')
    parser.add_argument('--seed', '-s',
                        type=int,
                        help='Seed for randomness')

    parser.add_argument('--ego-config',
                        type=json.loads,
                        default={},
                        help='Config for the ego agent')

    parser.add_argument('--alt-config',
                        type=json.loads,
                        default={},
                        help='Config for the partner agent')

    parser.add_argument('--env-config',
                        type=json.loads,
                        default={},
                        help='Config for the environment')

    # Wrappers
    parser.add_argument('--framestack', '-f',
                        type=int,
                        default=1,
                        help='Number of observations to stack')

    parser.add_argument('--record', '-r',
                        help='Saves joint trajectory into file specified')

    parser.add_argument('--render',
                        action='store_true',
                        help='Render the environment as it is being run')

    parser.add_argument('--ego-load',
                        help='File to load the ego agent from')
    parser.add_argument('--alt-load',
                        help='File to load the partner agent from')

    args = parser.parse_args()

    input_check(args)

    print(f"Arguments: {args}")
    env, altenv = generate_env(args)
    print(f"Environment: {env}; Partner env: {altenv}")
    ego = generate_agent(env, args.ego, args.ego_config, args.ego_load)
    print(f'Ego: {ego}')
    alt = generate_agent(altenv, args.alt, args.alt_config, args.alt_load)
    env.add_partner_agent(alt)
    print(f'Alt: {alt}')

    run_test(ego, env, args.total_episodes, args.render)

    if args.record is not None:
        env.get_transitions().write_transition(args.record)
