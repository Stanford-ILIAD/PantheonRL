import argparse
import json
import gym

import torch as th

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from pantheonrl.common.wrappers import frame_wrap, recorder_wrap
from pantheonrl.common.agents import OnPolicyAgent, StaticPolicyAgent

from pantheonrl.algos.adap.adap_learn import ADAP
from pantheonrl.algos.adap.policies import AdapPolicyMult, AdapPolicy
from pantheonrl.algos.adap.agent import AdapAgent

from pantheonrl.algos.modular.learn import ModularAlgorithm
from pantheonrl.algos.modular.policies import ModularPolicy

from pantheonrl.algos.bc import BCShell, reconstruct_policy

from pantheonrl.envs.rpsgym.rps import RPSEnv, RPSWeightedAgent
from pantheonrl.envs.blockworldgym import simpleblockworld, blockworld
from pantheonrl.envs.liargym.liar import LiarEnv, LiarDefaultAgent

from overcookedgym.overcooked_utils import LAYOUT_LIST

ENV_LIST = ['RPS-v0', 'BlockEnv-v0', 'BlockEnv-v1', 'LiarsDice-v0',
            'OvercookedMultiEnv-v0']

ADAP_TYPES = ['ADAP', 'ADAP_MULT']
EGO_LIST = ['PPO', 'ModularAlgorithm', 'LOAD'] + ADAP_TYPES
PARTNER_LIST = ['PPO', 'DEFAULT', 'FIXED'] + ADAP_TYPES


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

    # Construct alt configs
    if args.alt_config is None:
        args.alt_config = [{} for _ in args.alt]
    elif len(args.alt_config) != len(args.alt):
        raise EnvException(
            "Number of partners is different from number of configs")

    # Construct ego config
    if 'verbose' not in args.ego_config:
        args.ego_config['verbose'] = 1

    if (args.tensorboard_log is not None) != \
            (args.tensorboard_name is not None):
        raise EnvException("Must define log and names for tensorboard")


def latent_check(args):
    # Check for ADAP
    all_adap = all([v in ADAP_TYPES for v in args.alt])
    if args.ego not in ADAP_TYPES or not all_adap:
        raise EnvException(
            "both agents must be ADAP or ADAP_MULT to share latent spaces")

    if 'context_size' not in args.ego_config:
        args.ego_config['context_size'] = 3
    if 'context_sampler' not in args.ego_config:
        args.ego_config['context_sampler'] = "l2"

    for conf in args.alt_config:
        if 'context_size' not in conf:
            conf['context_size'] = args.ego_config['context_size']
        elif conf['context_size'] != args.ego_config['context_size']:
            raise EnvException("both agents must have similar configs \
                                to share latent spaces")

        if 'context_sampler' not in conf:
            conf['context_sampler'] = args.ego_config['context_sampler']
        elif conf['context_sampler'] != args.ego_config['context_sampler']:
            raise EnvException("both agents must have similar configs \
                                to share latent spaces")


def generate_env(args):
    env = gym.make(args.env, **args.env_config)

    altenv = env.getDummyEnv(1)

    if args.framestack > 1:
        env = frame_wrap(env, args.framestack)
        altenv = frame_wrap(altenv, args.framestack)

    if args.record is not None:
        env = recorder_wrap(env)

    return env, altenv


def generate_ego(env, args):
    kwargs = args.ego_config
    kwargs['env'] = env
    kwargs['device'] = args.device
    if args.seed is not None:
        kwargs['seed'] = args.seed

    kwargs['tensorboard_log'] = args.tensorboard_log

    if args.ego == 'LOAD':
        model = gen_load(kwargs, kwargs['type'], kwargs['location'])
        # wrap env in Monitor and VecEnv wrapper
        vec_env = DummyVecEnv([lambda: Monitor(env)])
        model.set_env(vec_env)
        if kwargs['type'] == 'ModularAlgorithm':
            model.policy.do_init_weights(init_partner=True)
            model.policy.num_partners = len(args.alt)
        return model
    elif args.ego == 'PPO':
        return PPO(policy='MlpPolicy', **kwargs)
    elif args.ego == 'ADAP':
        return ADAP(policy=AdapPolicy, **kwargs)
    elif args.ego == 'ADAP_MULT':
        return ADAP(policy=AdapPolicyMult, **kwargs)
    elif args.ego == 'ModularAlgorithm':
        policy_kwargs = dict(num_partners=len(args.alt))
        return ModularAlgorithm(policy=ModularPolicy,
                                policy_kwargs=policy_kwargs,
                                **kwargs)
    else:
        raise EnvException("Not a valid policy")


def gen_load(config, policy_type, location):
    if policy_type in ADAP_TYPES:
        if 'latent_val' not in config:
            raise EnvException("latent_val needs to be specified for \
                                FIXED ADAP policy")
        latent_val = th.tensor(config.pop('latent_val'))
        agent = ADAP.load(location)
        agent.policy.set_context(latent_val)
    elif policy_type == 'PPO':
        agent = PPO.load(location)
    elif policy_type == 'ModularAlgorithm':
        agent = ModularAlgorithm.load(location)
    elif policy_type == 'BC':
        agent = BCShell(reconstruct_policy(location))
    else:
        raise EnvException("Not a valid FIXED/LOAD policy")

    return agent


def gen_fixed(config, policy_type, location):
    agent = gen_load(config, policy_type, location)
    return StaticPolicyAgent(agent.policy)


def gen_default(config, altenv):
    if isinstance(altenv, RPSEnv):
        return RPSWeightedAgent(**config)

    if config:
        raise EnvException("No config possible for this default agent")

    if altenv == simpleblockworld.PartnerEnv:
        return simpleblockworld.SBWDefaultAgent()
    elif altenv == blockworld.PartnerEnv:
        return blockworld.DefaultConstructorAgent()
    elif isinstance(altenv, LiarEnv):
        return LiarDefaultAgent()
    else:
        raise EnvException("No default policy available")


def gen_partner(type, config, altenv, ego, args):
    if type == 'FIXED':
        return gen_fixed(config, config['type'], config['location'])
    elif type == 'DEFAULT':
        return gen_default(config, altenv)

    if args.tensorboard_log is not None:
        agentarg = {
            'tensorboard_log': args.tensorboard_log,
            'tb_log_name': args.tensorboard_name+'_alt_'+str(args.partner_num)
        }
    else:
        agentarg = {}

    config['env'] = altenv
    config['device'] = args.device
    if args.seed is not None:
        config['seed'] = args.seed
    config['verbose'] = args.verbose_partner

    if type == 'PPO':
        return OnPolicyAgent(PPO(policy='MlpPolicy', **config), **agentarg)

    if type == 'ADAP':
        alt = ADAP(policy=AdapPolicy, **config)
    elif type == 'ADAP_MULT':
        alt = ADAP(policy=AdapPolicyMult, **config)
    else:
        raise EnvException("Not a valid policy")

    shared = ego.policy if args.share_latent else None
    return AdapAgent(alt, latent_syncer=shared, **agentarg)


def generate_partners(altenv, env, ego, args):
    partners = []
    for i in range(len(args.alt)):
        args.partner_num = i
        v = gen_partner(args.alt[i],
                        args.alt_config[i],
                        altenv,
                        ego,
                        args)
        print(f'Partner {i}: {v}')
        env.add_partner_agent(v)
        partners.append(v)
    return partners


def preset(args, preset_id):
    '''
    helpful defaul configuration settings
    '''

    if preset_id == 1:
        env_name = args.env
        if 'layout_name' in args.env_config:
            env_name = "%s-%s" % (args.env, args.env_config['layout_name'])

        if args.tensorboard_log is None:
            args.tensorboard_log = 'logs'
        if args.tensorboard_name is None:
            args.tensorboard_name = '%s-%s%s-%d' % (
                env_name, args.ego, args.alt[0], args.seed)
        if args.ego_save is None:
            args.ego_save = 'models/%s-%s-ego-%d' % (
                env_name, args.ego, args.seed)
        if args.alt_save is None:
            args.alt_save = 'models/%s-%s-alt-%d' % (
                env_name, args.alt[0], args.seed)
        # if not args.record:
        #     args.record = 'trajs/%s-%s%s-%d' % (env_name, args.ego, args.alt[0], args.seed)
    else:
        raise Exception("Invalid preset id")
    return args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='''\
            Train ego and partner(s) in an environment.

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

            Alt-Agents:
            -----------
            The alt-agents are the partner agents that are embedded in the
            environment. If multiple are listed, the environment randomly
            samples one of them to be the partner at the start of each episode.

            Supported alt-agent algorithms include PPO, ADAP, ADAP_MULT,
            DEFAULT, and FIXED. DEFAULT refers to the default hand-made policy
            in the environment (if it exists). FIXED refers to a policy that
            has already been saved to a file, and will not learn anymore.

            Default parameters for these algorithms can be overriden using
            --alt-config. For FIXED policies, one must have parameters for
            `type` and `location` to load in the policies. If the FIXED
            policy is an ADAP policy, it must also have a `latent_val`
            parameter.

            NOTE:
            All configs are based on the json format, and will be interpreted
            as dictionaries for the kwargs of their initializers.

            Example usage (Overcooked with ADAP agents that share the latent
            space):

            python3 trainer.py OvercookedMultiEnv-v0 ADAP ADAP --env-config
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
                        nargs='+',
                        help='Algorithm for the partner agent')

    parser.add_argument('--total-timesteps', '-t',
                        type=int,
                        default=500000,
                        help='Number of time steps to run (ego perspective)')

    parser.add_argument('--device', '-d',
                        default='auto',
                        help='Device to run pytorch on')
    parser.add_argument('--seed', '-s',
                        default=None,
                        type=int,
                        help='Seed for randomness')

    parser.add_argument('--ego-config',
                        type=json.loads,
                        default={},
                        help='Config for the ego agent')

    parser.add_argument('--alt-config',
                        type=json.loads,
                        nargs='*',
                        help='Config for the ego agent')

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

    parser.add_argument('--ego-save',
                        help='File to save the ego agent into')
    parser.add_argument('--alt-save',
                        help='File to save the partner agent into')

    parser.add_argument('--share-latent', '-l',
                        action='store_true',
                        help='True when both actors are ADAP and want to sync \
                        latent values')

    parser.add_argument('--tensorboard-log',
                        help='Log directory for tensorboard')

    parser.add_argument('--tensorboard-name',
                        help='Name for ego in tensorboard')

    parser.add_argument('--verbose-partner',
                        action='store_true',
                        help='True when partners should log to output')

    parser.add_argument('--preset', type=int, help='Use preset args')

    args = parser.parse_args()

    if args.preset:
        args = preset(args, args.preset)
    input_check(args)

    if args.share_latent:
        latent_check(args)

    print(f"Arguments: {args}")
    env, altenv = generate_env(args)
    print(f"Environment: {env}; Partner env: {altenv}")
    ego = generate_ego(env, args)
    print(f'Ego: {ego}')
    partners = generate_partners(altenv, env, ego, args)

    learn_config = {'total_timesteps': args.total_timesteps}
    if args.tensorboard_log:
        learn_config['tb_log_name'] = args.tensorboard_name
    ego.learn(**learn_config)

    if args.record:
        transition = env.get_transitions()
        transition.write_transition(args.record)

    if args.ego_save:
        ego.save(args.ego_save)
    if args.alt_save:
        if len(partners) == 1:
            try:
                partners[0].model.save(args.alt_save)
            except AttributeError:
                print("FIXED or DEFAULT partners are not saved")
        else:
            for i in range(len(partners)):
                try:
                    partners[i].model.save(f"{args.alt_save}/{i}")
                except AttributeError:
                    print("FIXED or DEFAULT partners are not saved")
