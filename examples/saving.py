import pickle

import numpy as np
import tensorflow as tf

from tfrbm import BBRBM


def main():
    rbm_1 = BBRBM(n_visible=32, n_hidden=64)
    x = tf.convert_to_tensor(np.random.random((1, 32)), dtype=tf.float32)

    with open('rbm.pickle', 'wb') as f:
        pickle.dump(rbm_1, f)

    with open('rbm.pickle', 'rb') as f:
        rbm_2 = pickle.load(f)

    y_1 = rbm_1.compute_hidden(x)
    y_2 = rbm_2.compute_hidden(x)

    assert np.allclose(y_1.numpy(), y_2.numpy())


if __name__ == '__main__':
    main()
