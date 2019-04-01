""" Tests for logging utilities """
import tensorflow as tf

from deeprl import log_utils

def test_hidden_sizes_to_str_smoke():
    """ smoke test converting hidden sizes to str """
    assert '32' == log_utils.hidden_sizes_to_str((32,))
    assert '32-64' == log_utils.hidden_sizes_to_str((32, 64))

def test_kwargs_to_exp_name_strs():
    kwargs = {'num_runs': 3, 'activation': tf.tanh, 'hidden_sizes': (32, 64)}
    ret = log_utils.kwargs_to_exp_name_strs(kwargs)
    assert ret == ['tanh', 'hid32-64']

def test_kwargs_to_exp_name():
    ret = log_utils.kwargs_to_exp_name('test_exp', 0, {'hidden_sizes': (32,)})
    assert ret == 'test_exp_hid32_s0'
    ret = log_utils.kwargs_to_exp_name('test_exp', 0, {})
    assert ret == 'test_exp_s0'
    ret = log_utils.kwargs_to_exp_name(
            'test', 0, {'env_name': 'cartpole-v0', 'hidden_sizes': (64,64),
                        'activation': tf.nn.relu})
    assert ret == 'test_cartpole-v0_hid64-64_relu_s0'
