import numpy as np
import tensorflow as tf
import random
import functools
import scipy.signal
import time

"""
Some utility functions
"""

def timetest(input_func):

    def timed(*args, **kwargs):

        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        """
        print("Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}".format(
                input_func.__name__,
                args,
                kwargs,
                end_time - start_time
                ))
        """
        print("Method Name - {0}, Execution Time - {1}".format(
                input_func.__name__,
                end_time - start_time
                ))
        return result
    return timed

## a decorator to allow tf graphs to be split in to sub procs
## source code https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if no arguments are provided. All arguments must be optional.
    """
    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = '_cache_' + function.__name__
    name = scope or function.__name__
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            ## allow the variable scope to be skipped in some cases
            if name != 'skip':
                with tf.variable_scope(name, reuse=tf.AUTO_REUSE, *args, **kwargs):
                    setattr(self, attribute, function(self))
            else:
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator
 
## from DeepRL-Agents/A3C-Doom.ipynb           
# Discounting function used to calculate discounted returns.
def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

        
#Used to initialize weights for policy and value output layers
def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer        
        
        
def make_train_op(local_estimator, global_estimator):
    """
    Creates an op that applies local estimator gradients
    to the global estimator.
    """
    local_grads, _ = zip(*local_estimator.grads_and_vars)
    # Clip gradients
    local_grads, _ = tf.clip_by_global_norm(local_grads, 5.0)
    _, global_vars = zip(*global_estimator.grads_and_vars)
    local_global_grads_and_vars = list(zip(local_grads, global_vars))
    return global_estimator.optimizer.apply_gradients(local_global_grads_and_vars,
                                                      global_step=tf.train.get_global_step())
       