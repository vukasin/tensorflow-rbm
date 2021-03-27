import abc
import logging
from typing import Dict, List

import tensorflow as tf

from .util import xavier_init


class RBM(abc.ABC):
    def __init__(
            self,
            n_visible: int,
            n_hidden: int,
            learning_rate: float = 0.01,
            momentum: float = 0.95,
            xavier_const: float = 1.0
    ):
        """
        Initializes RBM.

        :param n_visible: number of visible neurons (input size)
        :param n_hidden: number of hidden neurons
        :param learning_rate: learning rate (default: 0.01)
        :param momentum: momentum (default: 0.95)
        :param xavier_const: constant used to initialize weights (default: 1.0)
        """
        if not 0.0 <= momentum <= 1.0:
            raise ValueError('momentum should be in range [0, 1]')

        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.w = tf.Variable(xavier_init(self.n_visible, self.n_hidden, const=xavier_const), dtype=tf.float32)
        self.visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.delta_w = tf.Variable(tf.zeros([self.n_visible, self.n_hidden]), dtype=tf.float32)
        self.delta_visible_bias = tf.Variable(tf.zeros([self.n_visible]), dtype=tf.float32)
        self.delta_hidden_bias = tf.Variable(tf.zeros([self.n_hidden]), dtype=tf.float32)

        self.logger = logging.getLogger(self.__class__.__name__)

    @abc.abstractmethod
    def step(self, x: tf.Tensor) -> tf.Tensor:
        """
        Performs one training step.

        :param x: tensor of shape (batch_size, n_visible)
        :return: tensor of size (1,) containing reconstruction error
        """
        raise NotImplementedError('step is not implemented')

    def _apply_momentum(self, old: tf.Tensor, new: tf.Tensor) -> tf.Tensor:
        n = tf.cast(new.shape[0], dtype=tf.float32)
        m = self.momentum
        lr = self.learning_rate

        return tf.add(tf.math.scalar_mul(m, old), tf.math.scalar_mul((1 - m) * lr / n, new))

    def compute_hidden(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes hidden state from the input.

        :param x: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_hidden)
        """
        return tf.nn.sigmoid(tf.matmul(x, self.w) + self.hidden_bias)

    def compute_visible(self, hidden: tf.Tensor) -> tf.Tensor:
        """
        Computes visible state from hidden state.

        :param hidden: tensor of shape (batch_size, n_hidden)
        :return: tensor of shape (batch_size, n_visible)
        """
        return tf.nn.sigmoid(tf.matmul(hidden, tf.transpose(self.w)) + self.visible_bias)

    def reconstruct(self, x: tf.Tensor) -> tf.Tensor:
        """
        Computes visible state from the input. Reconstructs data.

        :param x: tensor of shape (batch_size, n_visible)
        :return: tensor of shape (batch_size, n_visible)
        """
        return self.compute_visible(self.compute_hidden(x))

    def fit(
            self,
            dataset: tf.data.Dataset,
            epoches: int = 10,
            batch_size: int = 10
    ) -> List[float]:
        """
        Trains model.

        :param dataset: TensorFlow dataset
        :param epoches: number of epoches (default: 10)
        :param batch_size: batch size (default: 10)
        :return: list of batch errors
        """
        assert epoches > 0, "Number of epoches must be positive"

        errors = []

        for epoch in range(epoches):
            self.logger.info('Starting epoch %d', epoch)

            epoch_err_sum = 0.0
            epoch_err_num = 0

            for batch in dataset.batch(batch_size):
                err_t = self.step(batch)
                err_f: float = err_t.numpy().item()
                self.logger.debug('Batch error: %f', err_f)
                errors.append(err_f)
                epoch_err_sum += err_f
                epoch_err_num += 1

            self.logger.info('Epoch error: %f', epoch_err_sum / epoch_err_num)

        return errors

    def get_state(self) -> Dict[str, tf.Variable]:
        """
        Returns state of the model.

        :return: dictionary of TensorFlow variables
        """
        return {
            'w': self.w,
            'visible_bias': self.visible_bias,
            'hidden_bias': self.hidden_bias,
            'delta_w': self.delta_w,
            'delta_visible_bias': self.delta_visible_bias,
            'delta_hidden_bias': self.delta_hidden_bias
        }

    def set_state(self, state: Dict[str, tf.Variable]) -> None:
        """
        Restores model state.

        :param state: dictionary of TensorFlow variables
        :return: None
        """
        self.w = state['w']
        self.visible_bias = state['visible_bias']
        self.hidden_bias = state['hidden_bias']
        self.delta_w = state['delta_w']
        self.delta_visible_bias = state['delta_visible_bias']
        self.delta_hidden_bias = state['delta_hidden_bias']
