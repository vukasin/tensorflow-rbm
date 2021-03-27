import logging

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tfrbm import BBRBM


def transform_dataset(rbm, dataset):
    transformed_batches = []

    for batch in dataset.batch(2048):
        transformed_batches.append(rbm.compute_hidden(batch))

    return tf.data.Dataset.from_tensor_slices(tf.concat(transformed_batches, axis=0))


def main():
    logging.basicConfig(level=logging.INFO)

    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    x_train = x_train / 255
    x_test = x_test / 255

    dataset = tf.data.Dataset.from_tensor_slices(x_train.reshape(-1, 28 * 28).astype(np.float32))
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

    rbm_1 = BBRBM(n_visible=28 * 28, n_hidden=64)
    rbm_2 = BBRBM(n_visible=64, n_hidden=32)
    rbm_3 = BBRBM(n_visible=32, n_hidden=2)

    epoches = 100
    batch_size = 10

    rbm_1.fit(dataset, epoches=epoches, batch_size=batch_size)
    dataset_2 = transform_dataset(rbm_1, dataset)

    rbm_2.fit(dataset_2, epoches=epoches, batch_size=batch_size)
    dataset_3 = transform_dataset(rbm_2, dataset_2)

    rbm_3.fit(dataset_3, epoches=epoches, batch_size=batch_size)

    def encode(x):
        hidden_1 = rbm_1.compute_hidden(x)
        hidden_2 = rbm_2.compute_hidden(hidden_1)
        hidden_3 = rbm_3.compute_hidden(hidden_2)

        return hidden_3

    dataset_test = tf.data.Dataset.from_tensor_slices(x_test.reshape(-1, 28 * 28).astype(np.float32))
    encoded_test = []

    for batch in dataset_test.batch(2048):
        batch_encoded_tensor = encode(batch)
        batch_encoded = batch_encoded_tensor.numpy()
        encoded_test.append(batch_encoded)

    encoded_test = np.vstack(encoded_test)
    plt.scatter(encoded_test[:, 0], encoded_test[:, 1], alpha=0.5)
    plt.savefig('example.png')


if __name__ == '__main__':
    main()
