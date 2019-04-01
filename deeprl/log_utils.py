""" Logging utilities """


def kwargs_to_exp_name_strs(kwargs):
    """ given the kwargs to run, produce log dir name postfix """
    ignore = {'num_runs', 'epochs', 'steps_per_epoch', 'num_cpu'}
    shortnames = {'hidden_sizes': 'hid', 'activation': '', 'env_name': ''}
    str_val_funcs = {'hidden_sizes': hidden_sizes_to_str,
                     'activation': tf_activation_to_str}
    return [shortnames.get(name, name)
            + str_val_funcs.get(name, lambda x: str(x).lower())(val)
            for name, val in kwargs.items() if name not in ignore]


def hidden_sizes_to_str(hidden_sizes):
    """ convert hidden sizes tuple (32,64) and similar to str for exp name """
    return '-'.join(map(str, hidden_sizes))


def tf_activation_to_str(activation):
    """ convert tf activation function to string """
    return str(activation).split()[1]


def kwargs_to_exp_name(prefix, seed, kwargs):
    """ produce experiment name from run kwargs """
    exp_name_strs = kwargs_to_exp_name_strs(kwargs)
    after_prefix = '_' if exp_name_strs else ''
    return prefix + after_prefix + '_'.join(exp_name_strs) + '_s{}'.format(seed)
