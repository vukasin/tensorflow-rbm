import numpy as np
import tensorflow as tf


def xavier_init(fan_in, fan_out, *, const=1.0, dtype=tf.dtypes.float32):
    k = const * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval=-k, maxval=k, dtype=dtype)


def sample_bernoulli(ps):
    return tf.nn.relu(tf.sign(ps - tf.random.uniform(tf.shape(ps))))


def sample_gaussian(x, sigma):
    return x + tf.random.uniform(tf.shape(x), mean=0.0, stddev=sigma, dtype=tf.float32)
