import tensorflow as tf
import numpy as np
import logging
from PIL import Image

from tfrbm import BBRBM


def main():
    logging.basicConfig(level=logging.INFO)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    dataset = tf.data.Dataset.from_tensor_slices(x_train.reshape(-1, 28 * 28).astype(np.float32))
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

    rbm = BBRBM(n_visible=28 * 28, n_hidden=64)
    rbm.fit(dataset, epoches=100, batch_size=10)

    for i in np.random.choice(np.arange(x_test.shape[0]), 5, replace=False):
        x = x_test[i]
        x_tensor = tf.convert_to_tensor(x.reshape(1, 28 * 28), dtype=tf.float32)
        x_reconstructed_tensor = rbm.reconstruct(x_tensor)
        x_reconstructed = x_reconstructed_tensor.numpy().reshape(28, 28)

        Image.fromarray((x * 255).astype(np.uint8)).save(f'{i}_original.png')
        Image.fromarray((x_reconstructed * 255).astype(np.uint8)).save(f'{i}_reconstructed.png')


if __name__ == '__main__':
    main()
