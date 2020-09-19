import numpy as np


# Activation Functions

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


sigmoid_s = "def sigmoid(x):\n\treturn 1 / (1 + np.exp(-x))"


def sigmoid_prime(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


sigmoid_prime_s = "def sigmoid_prime(x):\n\tfx = sigmoid(x)\n\treturn fx * (1 - fx)"


# Hyperbolic Tangent
def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))


tanh_s = "def tanh(x):\n\treturn (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))"


def tanh_prime(x):
    t = tanh(x)
    return 1 - t ** 2


tanh_prime_s = "def tanh_prime(x):\n\tt = tanh(x)\n\treturn 1 - t ** 2"


# ReLU
def relu(x):
    return max(0, x)


relu_s = "def relu(x):\n\treturn max(0, x)"


def relu_prime(x):
    return 1 if x > 0 else 0


relu_prime_s = "def relu_prime(x):\n\treturn 1 if x > 0 else 0"


# Leaky/Parametric ReLU
def parametric_relu(x, alpha=0.01):
    return max(alpha * x, x)


parametric_relu_s = "def parametric_relu(x, alpha=0.01):\n\treturn max(alpha * x, x)"


def parametric_relu_prime(x, alpha=0.01):
    return 1 if x > 0 else alpha


parametric_relu_prime_s = "def parametric_relu_prime(x, alpha=0.01):\n\treturn 1 if x > 0 else alpha"


# ELU
def elu(x, alpha):
    return x if x >= 0 else alpha * (np.exp(x) - 1)


elu_s = "def elu(x, alpha):\n\treturn x if x >= 0 else alpha * (np.exp(x) - 1)"


def elu_prime(x, alpha):
    return 1 if x > 0 else alpha * np.exp(x)


elu_prime_s = "def elu_prime(x, alpha):\n\treturn 1 if x > 0 else alpha * np.exp(x)"


# Swish
def swish(x):
    return x / (1 + np.exp(-x))


swish_s = "def swish(x):\n\treturn x / (1 + np.exp(-x))"


def swish_prime(x):
    sig = sigmoid(x)
    swi = swish(x)
    return swi + (sig * (1 - swi))


swish_prime_s = "def swish_prime(x):\n\tsig = sigmoid(x)\n\tswi = swish(x)\n\treturn swi + (sig * (1 - swi))"


# Linear
def linear(x, slope):
    return x * slope


linear_s = "def linear(x, slope):\n\treturn x * slope"


def linear_prime(slope):
    return slope


linear_prime_s = "def linear_prime(slope):\n\treturn slope"


# Binary Step
def binary(x):
    return 1 if x > 0 else 0


binary_s = "def binary(x):\n\treturn 1 if x > 0 else 0"


def binary_prime():
    return 0


binary_prime_s = "def binary_prime():\n\treturn 0"


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()


mse_loss_s = "def mse_loss(y_true, y_pred):\n\treturn ((y_true - y_pred) ** 2).mean()"


class Neuron:
    def __init__(self, number_of_weights, activation="sigmoid"):
        self.number_of_weights = number_of_weights
        self.input_vec = 0
        self.activation_val = 0
        self.weights = np.random.rand(number_of_weights)
        self.biases = np.random.rand()
        self.learn_rates = None
        self.weight_partials = None
        self.node_partials = None
        self.bias_partial = None

    def feedforward(self, inputs):
        # Weight inputs, add bias, and use activation function
        self.input_vec = inputs
        self.sum_val = np.dot(self.weights, inputs) + self.biases
        # FIXME self.activation_val = activation(self.sum_val)
        self.sigmoid_val = sigmoid(self.sum_val)
        return self.sigmoid_val

    def get_weight(self, index):
        return self.weights[index]

    def get_weights(self):
        return self.weights

    def set_weight(self, index, value):
        self.weights[index] = value

    # FIXME
    def calculate_deriv(self):
        return 0



class Layer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = []
        self.output_vec = None

    def get_num_nodes(self):
        return self.num_nodes

    def get_nodes(self):
        return self.nodes

    def append_node(self, node):
        self.nodes.append(node)


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.output_vec = []

    def feedforward_layer(self, layer, x_vec):  # layer = layers(i)
        output = np.array([])
        for node in layer.nodes:
            node_output = node.feedforward(x_vec)
            output = np.append(output, node_output)
        return output

    def feedforward_network(self, input_vec):
        self.inputs = input_vec
        output_vec = self.feedforward_layer(self.layers[1], input_vec)
        for layerIdx in range(2, len(self.layers)):
            output_vec = self.feedforward_layer(self.layers[layerIdx], output_vec)
        self.output_vec = output_vec


    def update_node_weights(self, node, node_number, layer_number):
        #node.weights -= self.d_l_dy_pred * node.node_partials * node.weight_partials
        inputs = node.input_vec
        #FIXME
        if layer_number == len(self.layers) - 1:
            derivatives = np.array([])
            for input in inputs:
                deriv = sigmoid_prime(input)
                derivatives = np.append(derivatives, deriv)
                #while len(error_temp) != self.layers[-1].num_nodes:
                    #pass
            delta_w = self.error[node_number] * inputs * derivatives
            node.weights += delta_w
        else:
            pass


    def backprop_node(self, node, deriv):
        # FIXME
        # pred = node.sigmoid_val
        node.bias_partial = deriv
        node.weight_partials = np.array(node.input_vec) * deriv
        node.node_partials = node.get_weights() * deriv
        return

    def backprop_layer(self, layer):
        for node in layer.get_nodes():
            # FIXME
            deriv = sigmoid_prime(node.sum_val)
            self.backprop_node(node, deriv)
        return


    # where all the weights get updated
    def backprop_network(self, data, true, learn_rate, cycles):
        #self.d_l_dy_pred = -2 * (true - self.output_vec) # size 3
        for i in range(0, cycles):
            self.feedforward_network(data)
            self.error = true - self.output_vec
            print(self.error)
            #self.learning_multiplier = learn_rate * self.d_l_dy_pred
            for layerIdx in range(len(self.layers) - 1, 0, -1):
                current_layer = self.layers[layerIdx]
                self.backprop_layer(current_layer)
                for nodeIdx in range(0, current_layer.get_num_nodes()): # output: loops 012 # next in: loops 0123
                    self.update_node_weights(current_layer.nodes[nodeIdx], nodeIdx, layerIdx) # this works only for output layer



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
            layer.append_node(Neuron(layers[layerIdx - 1].get_num_nodes()))


def network(inputs, outputs, nodes_per_layer, activation):
    layers = create_layers(inputs, outputs, nodes_per_layer)
    create_nodes(layers, activation)
    network = NeuralNetwork(layers)
    network.feedforward_network(np.array([-3.0, -3.0]))
    data = np.array([1, 2])
    true = np.array([.7])
    learn_rate = 0.1
    cycles = 10
    network.backprop_network(data, true, learn_rate, cycles)
    print(network.output_vec)


#network(inputs=3, outputs=1, nodes_per_layer=(2,), activation="sigmoid")
network(inputs=2, outputs=2, nodes_per_layer=(2,3), activation="sigmoid")


