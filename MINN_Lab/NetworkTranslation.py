import numpy as np


# Activation Functions

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid2 = "def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))"


def sigmoid_prime(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


# Hyperbolic Tangent
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


def tanh_prime(x):
    t = tanh(x)
    return 1 - t ** 2


# ReLU
def relu(x):
    return max(0, x)


def relu_prime(x):
    return 1 if x > 0 else 0


# Leaky/Parametric ReLU
def parametric_relu(x, alpha=0.01):
    return max(alpha * x, x)


def parametric_relu_prime(x, alpha=0.01):
    return 1 if x > 0 else alpha


# ELU
def elu(x, alpha):
    return x if x >= 0 else alpha * (np.exp(x) - 1)


def elu_prime(x, alpha):
    return 1 if x > 0 else alpha * np.exp(x)


# Swish
def swish(x):
    return x / (1 + np.exp(-x))


def swish_prime(x):
    sig = sigmoid(x)
    swi = swish(x)
    return swi + (sig * (1 - swi))


# Linear
def linear(x, slope):
    return x * slope


def linear_prime(slope):
    return slope


# Binary Step
def binary(x):
    return 1 if x > 0 else 0


def binary_prime():
    return 0


class Neuron:
    def __init__(self, number_of_weights, number_of_biases, activation):
        self.number_of_weights = number_of_weights
        self.number_of_biases = number_of_biases
        self.input_val = 0
        self.activation_val = 0
        self.weights = np.random.rand(number_of_weights)
        self.biases = np.random.rand(number_of_biases)

    def feedforward(self, inputs):
        # Weight inputs, add bias, and use activation function
        self.sum_val = np.dot(self.weights, inputs) + self.biases
        # FIXME self.activation_val = activation(self.sum_val)
        return

    def get_weight(self, index):
        return self.weights[index]

    def set_weight(self, index, value):
        self.weights[index] = value


class Layer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = []

    def get_num_nodes(self):
        return self.num_nodes

    def get_nodes(self):
        return self.nodes

    def append_node(self, node):
        self.nodes.append(node)


def create_layers(inputs, outputs, nodes_per_layer):
    layers = [Layer(inputs)]
    for i in range(0, len(nodes_per_layer)):
        layers.append(Layer(nodes_per_layer[i]))  # nodes_per_layer does not include nodes of input and output
    layers.append(Layer(outputs))
    return layers


def create_nodes(layers, activation):
    layer_length = len(layers)
    for layerIdx in range(1, layer_length):
        layer = layers[layerIdx]
        for i in range(0, layer.num_nodes):
            layer.append_node(Neuron(layers[layerIdx - 1].get_num_nodes(), layer.get_num_nodes()))


def network(inputs, outputs, nodes_per_layer, activation):
    layers = create_layers(inputs, outputs, nodes_per_layer)
    create_nodes(layers, activation)


network(inputs=2, outputs=3, nodes_per_layer=(2,), activation="bruh")
