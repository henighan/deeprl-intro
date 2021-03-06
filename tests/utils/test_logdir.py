""" Tests for logging utilities """
import tensorflow as tf

from deeprl.utils import logdir


def test_hidden_sizes_to_str_smoke():
    """ smoke test converting hidden sizes to str """
    assert logdir.hidden_sizes_to_str((32,)) == '32'
    assert logdir.hidden_sizes_to_str((32, 64)) == '32-64'


def test_kwargs_to_exp_name_strs():
    """ test converting kwargs to strings for exp name """
    kwargs = {'num_runs': 3, 'activation': tf.tanh, 'hidden_sizes': (32, 64),
              'algo': 'ppo'}
    ret = logdir.kwargs_to_exp_name_strs(kwargs)
    assert ret == ['tanh', 'hid32-64', 'ppo']


def test_kwargs_to_exp_name():
    """ test converting kwargs to exp name """
    ret = logdir.kwargs_to_exp_name('test_exp', {'hidden_sizes': (32,)})
    assert ret == 'test_exp_hid32'
    ret = logdir.kwargs_to_exp_name('test_exp', {})
    assert ret == 'test_exp'
    ret = logdir.kwargs_to_exp_name(
        'test', {'env_name': 'cartpole-v0', 'hidden_sizes': (64, 64),
                 'activation': tf.nn.relu})
    assert ret == 'test_cartpole-v0_hid64-64_relu'
    ret = logdir.kwargs_to_exp_name(
        '', {'env_name': 'cartpole-v0'})
    assert ret == 'cartpole-v0'


def test_output_dir_from_kwargs():
    """ test converting kwargs to output_dir """
    ret = logdir.output_dir_from_kwargs('', 'tom', {'hidden_sizes': (32,)})
    assert ret == './data/tom/hid32/hid32'
    ret = logdir.output_dir_from_kwargs('', 'tom', {'hidden_sizes': (32,)},
                                        seed=0)
    assert ret == './data/tom/hid32/hid32_s0'
