
def preset(args, preset_id):
    if preset_id == 1:
        if not args.tensorboard_log: args.tensorboard_log = 'logs'
        if not args.tensorboard_name: args.tensorboard_name = '%s-%s-%s%s-%d' % (args.env, args.env_config['layout_name'], args.ego, args.alt[0], args.seed)
        if not args.ego_save: args.ego_save = 'models/%s-%s-%s-ego-%d'  % (args.env, args.env_config['layout_name'], args.ego, args.seed)
        if not args.alt_save: args.alt_save = 'models/%s-%s-%s-alt-%d' % (args.env, args.env_config['layout_name'], args.alt[0], args.seed)
    else:
        raise Exception("Invalid preset id")
    return args