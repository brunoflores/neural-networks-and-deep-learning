"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feed-forward neural network.  Gradients are calculated
using back-propagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

# Libraries
# Standard library
import random

# Third-party libraries
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feed_forward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        training_data = list(training_data)
        n = len(training_data)
        n_test = None

        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using back-propagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b, nabla_w = self.back_propagate_matrix_based(mini_batch)

        # Update weights
        # nabla_w is list of two numpy arrays 30x784 and 10x30
        self.weights = [w - (eta / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        # Update biases
        # nabla_b is list of two numpy arrays 30x1 and 10x1
        self.biases = [b - (eta / len(mini_batch)) * np.asarray(nb)
                       for b, nb in zip(self.biases, nabla_b)]

    def back_propagate_matrix_based(self, mini_batch):
        """
        A Matrix-based approach for computing gradients for all training
        examples in a mini-batch simultaneously.

        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``.

        :param mini_batch: A list of tuples ``(x, y)``
        :return: A tuple ``(nabla_b, nabla_w)``
        """
        x = np.asarray([x for x, y in mini_batch]).transpose()[0]
        y = np.asarray([y for x, y in mini_batch]).transpose()[0]

        zs, activations = self.forward_pass(x)
        delta, nabla_b, nabla_w = self.output_error(zs, activations, y)
        nabla_b, nabla_w = self.backward_pass(delta, zs, activations, nabla_b, nabla_w)

        # Sum the deltas of every neuron in each layer
        nabla_b = [np.sum(delta, axis=1, keepdims=True) for delta in nabla_b]

        return nabla_b, nabla_w

    def forward_pass(self, x):
        """
        Perform a forward pass through the net for inputs in the mini-match to
        compute a list of weighted inputs, ``zs``, for every layer z_l and
        a list of activations, ``activations``, for every layer a_l.

        It does this by summing all weight, w_l, multiplied by
        activations from the previous layer, a_l-1, plus a bias.
        Keep them because they will be used again to calculate the error, delta.

        The list of activations is just the weighted inputs with the
        activation function applied everywhere.

        :param x: A matrix whose columns are the vectors in the mini-batch
        :return: List of weighted inputs and list of activations
        """
        activations = [x]
        zs = []
        for b, w in zip(self.biases, self.weights):
            zs.append(np.dot(w, activations[-1]) + b)
            activations.append(sigmoid(zs[-1]))
        return zs, activations

    def output_error(self, zs, activations, y):
        """
        After a forward pass, compute the output error for every input in the mini-batch and
        back-propagate to the 2nd-last layer.

        Output error, delta, is here the derivative of C times the derivative of sigmoid using the
        weighted input for the last layer.
        Back-propagate to the previous layer by computing the dot product against the transposed activations
        from the previous layer.

        :param zs: List of weighted inputs
        :param activations: List of activations
        :param y: Ground-truth for inputs in the mini-batch
        :return: Output error, initial layer-by-layer lists of changes in biases and weights
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        delta = self.cost_derivative_for_output(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        return delta, nabla_b, nabla_w

    def backward_pass(self, delta, zs, activations, nabla_b, nabla_w):
        """
        Having the output error, perform a backward pass through the net calculating changes in biases and weights.

        Update the error, delta, about every layer using weights from the previous layer.
        Back-propagate until the end by computing the dot product against the transposed activations
        from the previous layer.

        :param delta: Output error
        :param zs: List of weighted inputs
        :param activations: List of activations
        :param nabla_b: Layer-by-layer list of changes in biases
        :param nabla_w: Layer-by-layer list of changes in weights
        :return: Final layer-by-layer lists of changes in biases and weights
        """
        for l in range(2, self.num_layers):
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sigmoid_prime(zs[-l])
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feed_forward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    @staticmethod
    def cost_derivative_for_output(output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return output_activations - y


# Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
