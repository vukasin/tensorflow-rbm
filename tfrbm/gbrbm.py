import tensorflow as tf

from .rbm import RBM
from .util import sample_bernoulli, sample_gaussian


class GBRBM(RBM):
    def __init__(
            self,
            n_visible: int,
            n_hidden: int,
            sample_visible: bool = False,
            sigma: float = 1.0,
            **kwargs
    ):
        """
        Initializes Gaussian-Bernoulli RBM.

        :param n_visible: number of visible neurons (input size)
        :param n_hidden: number of hidden neurons
        :param sample_visible: if reconstructed state should be sampled from the Gaussian distribution (default: False)
        :param sigma: standard deviation of this distribution, does nothing when sample_visible = False (default: 1.0)
        :param learning_rate: learning rate (default: 0.01)
        :param momentum: momentum (default: 0.95)
        :param xavier_const: constant used to initialize weights (default: 1.0)
        """
        self.sample_visible = sample_visible
        self.sigma = sigma

        super().__init__(n_visible, n_hidden, **kwargs)

    def step(self, x: tf.Tensor) -> tf.Tensor:
        hidden_p = tf.nn.sigmoid(tf.matmul(x, self.w) + self.hidden_bias)
        visible_recon_p = tf.matmul(sample_bernoulli(hidden_p), tf.transpose(self.w)) + self.visible_bias

        if self.sample_visible:
            visible_recon_p = sample_gaussian(visible_recon_p, self.sigma)

        hidden_recon_p = tf.nn.sigmoid(tf.matmul(visible_recon_p, self.w) + self.hidden_bias)

        positive_grad = tf.matmul(tf.transpose(x), hidden_p)
        negative_grad = tf.matmul(tf.transpose(visible_recon_p), hidden_recon_p)

        self.delta_w = self._apply_momentum(
            self.delta_w,
            positive_grad - negative_grad
        )
        self.delta_visible_bias = self._apply_momentum(
            self.delta_visible_bias,
            tf.reduce_mean(x - visible_recon_p, 0)
        )
        self.delta_hidden_bias = self._apply_momentum(
            self.delta_hidden_bias,
            tf.reduce_mean(hidden_p - hidden_recon_p, 0)
        )

        self.w.assign_add(self.delta_w)
        self.visible_bias.assign_add(self.delta_visible_bias)
        self.hidden_bias.assign_add(self.delta_hidden_bias)

        return tf.reduce_mean(tf.square(x - visible_recon_p))
