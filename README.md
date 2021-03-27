# tensorflow-rbm

Tensorflow implementation of Restricted Boltzmann Machine for layer-wise pretraining of deep autoencoders.

This is a fork of a [Michal Lukac repository](https://github.com/Cospel/rbm-ae-tf) with some corrections and improvements.

The Restricted Boltzmann Machine is a legacy machine learning model that is no longer used anywhere.
This repository is of historical and educational value only. I have updated the code using the [TensorFlow 2](https://www.tensorflow.org) to run on modern systems, 
but I will no longer maintain it.

### Installation

```shell
git clone https://github.com/meownoid/tensorfow-rbm.git
cd tensorfow-rbm
python -m pip install -r requirements.txt
python setup.py
```

### Example
Bernoulli-Bernoulli RBM is good for Bernoulli-distributed binary input data. MNIST, for example. To train the model, 
simply construct the `tf.data.Dataset` containing vectors of shape `(n_visible,)` and pass it to the `fit` method.

```python
import tensorflow as tf

from tfrbm import BBRBM


(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train / 255
x_test = x_test / 255

dataset = tf.data.Dataset.from_tensor_slices(x_train.reshape(-1, 28 * 28))
dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)

rbm = BBRBM(n_visible=28 * 28, n_hidden=64)
rbm.fit(dataset, epoches=100, batch_size=10)
```

Now you can use `reconstruct` method to check the model performance.

```python
# x_tensor: (1, n_visible)
# x_reconstructed_tensor: (1, n_visible)
x_reconstructed_tensor = rbm.reconstruct(x_tensor)
```

Full example code can be found in the `examples/mnist.py`.

Some examples of original and reconstructed data:

![MNIST Example](https://storage.yandexcloud.net/meownoid-pro-static/external/github/tensorflow-rbm/mnist-example.png)

### API

![API Diagram](https://storage.yandexcloud.net/meownoid-pro-static/external/github/tensorflow-rbm/api-scheme.png)

```python
BBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95)
```

```python
GBRBM(n_visible, n_hidden, learning_rate=0.01, momentum=0.95, sample_visible=False, sigma=1.0)
```

Initializes Bernoulli-Bernoulli RBM or Gaussian-Bernoulli RBM.

* `n_visible` — number of visible neurons (input size)
* `n_hidden` — number of hidden neurons

Only for `GBRBM`:

* `sample_visible` — sample reconstructed data with Gaussian distribution (with reconstructed value as a mean and a `sigma` parameter as deviation) or not (if not, every gaussoid will be projected into a single point)
* `sigma` — standard deviation of the input data

Advices:

* Use **BBRBM** for Bernoulli distributed data. Input values in this case must be in the interval from `0` to `1`.
* Use **GBRBM** for normally distributed data with `0` mean and `sigma` standard deviation. Normalize input data if necessary.

```python
rbm.fit(dataset, epoches=10, batch_size=10)
```

Trains the model and returns a list of errors.

* `dataset` — `tf.data.Dataset` composed of tensors of shape `(n_visible,)`
* `epoches` — number of epoches
* `batch_size` — batch size, should be as small as possible

```python
rbm.step(x)
```

Performs one training step and returns reconstruction error.

* `x` – tensor of shape `(batch_size, n_visible)`

```python
rbm.compute_hidden(x)
```

Computes hidden state from the input.

* `x` – tensor of shape `(batch_size, n_visible)`

```python
rbm.compute_visible(hidden)
```

Computes visible state from hidden state.

* `x` – tensor of shape `(batch_size, n_hidden)`

```python
rbm.reconstruct(x)
```

Computes visible state from the input. Reconstructs data.

* `x` – tensor of shape `(batch_size, n_visible)`

### Original README

Tensorflow implementation of Restricted Boltzman Machine and Autoencoder for layerwise pretraining of Deep Autoencoders with RBM. Idea is to first create RBMs for pretraining weights for autoencoder. Then weigts for autoencoder are loaded and autoencoder is trained again. In this implementation you can also use tied weights for autoencoder(that means that encoding and decoding layers have same transposed weights!).

I was inspired with these implementations but I need to refactor them and improve them. I tried to use also similar api as it is in [tensorflow/models](https://github.com/tensorflow/models):

> [myme5261314](https://gist.github.com/myme5261314/005ceac0483fc5a581cc)

> [saliksyed](https://gist.github.com/saliksyed/593c950ba1a3b9dd08d5)

> Thank you for your gists!

More about pretraining of weights in this paper:
> [Reducing the Dimensionality of Data with Neural Networks](https://www.cs.toronto.edu/~hinton/science.pdf)

Feel free to make updates, repairs. You can enhance implementation with some tips from:
> [Practical Guide to training RBM](https://www.cs.toronto.edu/~hinton/absps/guideTR.pdf)
