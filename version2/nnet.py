import numpy as np
import pandas as pd


#### Define the quadratic and cross-entropy cost functions

class Cost:

    @staticmethod
    def loss(z, v, pi, p, lmbda, theta):
        """Loss function to minimise to learn the neural network parameters.

        Arguments:
            z -- game outcome of a game of self-play
            v -- estimate of z from a position s (output of neural network from position s)
            pi -- numpy vector of search probabilities from MCTS
            p -- numpy vector of move probabilities (output of neural network from position s)
            c -- hyper-parameter controlling level of l2 weight regularisation
            theta -- numpy vector containing the parameters of neural network

        Returns:
            loss to be minimised
        """
        return (z - v)**2 - pi*np.log(p) + lmbda*np.sum(theta**2)

    @staticmethod
    def delta(a, y):
        """return the delta for backpropogation

        Arguments:
            a -- output from network, contains vector p and evaluation v
            y -- target for network, output from mcts, contains vector pi and game outcome z

        Returns:
            _description_
        """
        p, v = a[:-1], a[-1]
        pi, z = y[:-1], y[-1]
        out = pi*(1 - p)
        output = np.append(out, 2*(v-z)*v*(1-v), axis = 0)
        return output

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


#### Main Network class
class Network:

    def __init__(self, sizes, cost=Cost):
        """The list ``sizes`` contains the number of neurons in the respective
        layers of the network.  For example, if the list was [2, 3, 1]
        then it would be a three-layer network, with the first layer
        containing 2 neurons, the second layer 3 neurons, and the
        third layer 1 neuron.  The biases and weights for the network
        are initialized randomly, using
        ``self.initialise_weights`` (see docstring for that
        method).

        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.load_params()
        self.cost = cost

    def load_params(self):
        """Initialize each weight using a Gaussian distribution with mean 0
        and standard deviation 1 over the square root of the number of
        weights connecting to the same neuron.  Initialize the biases
        using a Gaussian distribution with mean 0 and standard
        deviation 1.

        Note that the first layer is assumed to be an input layer, and
        by convention we won't set any biases for those neurons, since
        biases are only ever used in computing the outputs from later
        layers.

        """
        try:
            df = pd.read_csv("network_params.csv", header=None)
            ind = df.shape[0]
            w = np.array(df.loc[ind-2])
            b = np.array(df.loc[ind-1])
            weights = w[np.logical_not(np.isnan(w))]
            biases = b[np.logical_not(np.isnan(b))]
            count = 0
            temp_b = []
            for y in self.sizes[1:]:
                temp_b.append(np.array(biases[count:count+y]).reshape([y,1]))
                count += y
            count = 0
            temp_w = []
            for x, y in zip(self.sizes[:-1], self.sizes[1:]):
                temp_w.append(np.array(weights[count:count+x*y]).reshape([y,x]))
                count += x*y
            self.biases = temp_b
            self.weights = temp_w
        except pd.errors.EmptyDataError:
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                        for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def predict(self, s):
        """Return the output of the network if ``s`` is input."""
        for b, w in zip(self.biases, self.weights):

            s = sigmoid(np.dot(w, s)+b.flatten())
        p = s[:-1]
        v = s[-1]
        return p, v

    def train_new_data(self, examples, eta, lmbda): #, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(x, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        print(len(examples))
        print(len(examples[0]))
        for x, y in examples:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda))*w-(eta/len(examples))*nw # lmbda/n
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(examples))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    # something weird about x and y here
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = (self.cost).delta(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def save_network_params(self):
        w = np.array(self.weights).flatten()
        b = np.array(self.biases).flatten()
        params = [list(w), list(b)]
        df = pd.DataFrame(params)
        df.to_csv("network_params.csv", mode="a", index=False, header=False)
