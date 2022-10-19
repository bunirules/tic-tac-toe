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
        # print("pi: ", pi)
        # print("p: ", p)
        out = pi*(1 - p)
        output = np.append(out, 2*(v-z)*v*(1-v))
        return output

def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))


#### Main Network class
class Network:

    def __init__(self, sizes, cost=Cost, load=False):
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
        self.load_params(load)
        self.cost = cost

    def load_params(self, load):
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
            if not load:
                raise pd.errors.EmptyDataError
            if load:
                df = pd.read_csv("network_params.csv", header=None)
                ind = df.shape[0]
                w = np.array(df.loc[ind-2])
                b = np.array(df.loc[ind-1])
                weights = w[np.logical_not(np.isnan(w))]
                biases = b[np.logical_not(np.isnan(b))]
                count = 0
                temp_b = []
                for y in self.sizes[1:]:
                    temp_b.append(np.array(biases[count:count+y]).reshape([y]))
                    count += y
                count = 0
                temp_w = []
                for x, y in zip(self.sizes[:-1], self.sizes[1:]):
                    temp_w.append(np.array(weights[count:count+x*y]).reshape([y,x]))
                    count += x*y
                self.biases = temp_b
                self.weights = temp_w
        except pd.errors.EmptyDataError:
            self.biases = [np.random.randn(y) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x)/np.sqrt(x) 
                    for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def predict(self, s):
        """Return the output of the network if ``s`` is input."""
        for b, w in zip(self.biases, self.weights):

            s = sigmoid(np.dot(w, s)+b.flatten())
        p = s[:-1]
        v = s[-1]
        return p, v

    def train_new_data(self, examples, eta, lmbda, n):
        """Update the network's weights and biases by applying gradient
        descent using backpropagation to a single mini batch.  The
        ``mini_batch`` is a list of tuples ``(s, y)``, ``eta`` is the
        learning rate, ``lmbda`` is the regularization parameter, and
        ``n`` is the total size of the training data set.

        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for example in examples:
            delta_nabla_b, delta_nabla_w = self.backprop(example[0], example[1])
            # for nb, dnb in zip(nabla_b, delta_nabla_b):
            #     print(f"nb: {nb},\n dnb: {dnb}")
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1-eta*(lmbda/n))*w-(eta/len(examples))*nw # lmbda/n
                        for w, nw in zip(self.weights, nabla_w)]
        # print("im here aksfazlkrgnszrf")
        # for b, nb in zip(self.biases, nabla_b):
        #     print(f"b: {b},\n nb: {nb}")
        # print("1, ",self.biases)
        self.biases = [b-(eta/len(examples))*nb
                       for b, nb in zip(self.biases, nabla_b)]
        # print(type(zip(self.biases, nabla_b)))
        # print("2, ",self.biases)

    # something weird about x and y here
    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        # print("x: ", x)
        # print("y: ", y)
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            # print("b: ", b.shape)
            # print(self.biases)
            # print("w: ", w.shape)
            # print("act: ", activation.shape)
            z = np.dot(w, activation)+b.flatten()
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        # print("act: ", activations[-1])
        delta = (self.cost).delta(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.outer(delta, activations[-2])
        # print("hello: ",nabla_w[-1].shape)
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
            # print("1234: ", delta.shape)
            nabla_b[-l] = delta
            nabla_w[-l] = np.outer(delta, activations[-l-1])
            # print("5678: ",nabla_b)
        return (nabla_b, nabla_w)

    def save_network_params(self):
        w = self.weights[0]
        for i in range(1, len(self.weights)):
            w = np.append(w, self.weights[i])
        b = self.biases[0]
        for i in range(1, len(self.biases)):
            b = np.append(b, self.biases[i])
        params = [list(w), list(b)]
        df = pd.DataFrame(params)
        df.to_csv("network_params.csv", mode="a", index=False, header=False)
