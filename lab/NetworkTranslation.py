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
    def __init__(self, number_of_weights):
        self.number_of_weights = number_of_weights
        self.input_vec = np.array([])
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

    def set_weight(self, index, value):
        self.weights[index] = value

    def backprop_node(self, deriv):
        # FIXME
        self.bias_partial = deriv
        self.weight_partials = np.array(self.input_vec) * deriv
        self.node_partials = self.weights * deriv


class Layer:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.nodes = []
        self.errors_vec = None

    def get_num_nodes(self):
        return self.num_nodes

    def get_nodes(self):
        return self.nodes

    def append_node(self, node):
        self.nodes.append(node)

    def feedforward_layer(self, x_vec):  # layer = layers(i)
        output = np.array([])
        for node in self.nodes:
            node_output = node.feedforward(x_vec)
            output = np.append(output, node_output)
        return output

    def backprop_layer(self):
        for node in self.get_nodes():
            # FIXME
            deriv = sigmoid_prime(node.sum_val)
            node.backprop_node(deriv)

    def layer_error(self, layer_in_front):
        errors = np.array([])
        for i in range(0, self.num_nodes):
            weights = np.array([])
            for node in layer_in_front.nodes:
                weights = np.append(weights, node.weights[i])
            errors = np.append(errors, np.dot(weights, layer_in_front.errors_vec))
        self.errors_vec = errors


class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers
        self.output_vec = np.array([])

    def feedforward_network(self, input_vec):
        self.inputs = input_vec
        output_vec = self.layers[1].feedforward_layer(input_vec)
        for layerIdx in range(2, len(self.layers)):
            output_vec = self.layers[layerIdx].feedforward_layer(output_vec)
        return output_vec

    def update_node_weights(self, node, node_number, layer_number, layer_in_front=0):
        inputs = node.input_vec
        # FIXME
        if layer_number == len(self.layers) - 1:
            derivatives = np.array([])
            for input in inputs:
                deriv = sigmoid_prime(input)
                derivatives = np.append(derivatives, deriv)
            delta_w = self.error[node_number] * inputs * derivatives
            node.weights += delta_w
        else:
            # dot product of derivatives of the output weights and the output errors
            scalars = np.array([])
            for i in range(0, self.layers[layer_number].num_nodes):
                weight_deriv = np.array([])
                for node in layer_in_front.nodes:
                    weight_deriv = np.append(weight_deriv, sigmoid_prime(node.weights[i]))
                scalars = np.append(scalars, np.dot(weight_deriv, layer_in_front.errors_vec))
            delta_w = node.input_vec * node.node_partials * scalars[node_number]
            node.weights += delta_w

    # where all the weights get updated
    def backprop_network(self, data, true):
        self.error = true - self.feedforward_network(data)
        self.layers[-1].errors_vec = self.error
        for layerIdx in range(len(self.layers) - 1, 0, -1):
            current_layer = self.layers[layerIdx]
            if layerIdx <= len(self.layers) - 2:
                current_layer.layer_error(self.layers[layerIdx + 1])
            current_layer.backprop_layer()
            for nodeIdx in range(0, current_layer.get_num_nodes()):  # output: loops 012 # next in: loops 0123
                if layerIdx <= len(self.layers) - 2:
                    current_layer.layer_error(self.layers[layerIdx + 1])
                    self.update_node_weights(current_layer.nodes[nodeIdx], nodeIdx,
                                             layerIdx, self.layers[layerIdx + 1])
                else:
                    self.update_node_weights(current_layer.nodes[nodeIdx], nodeIdx,
                                             layerIdx)

    def train(self, dataset, true_set, epoch=1000):
        for i in range(epoch):
            for data, true in zip(dataset, true_set):
                self.backprop_network(data, true)


"""
This is a comment
"""


def create_layers(inputs, outputs, nodes_per_layer):
    layers = [Layer(inputs)]
    for i in range(0, len(nodes_per_layer)):
        layers.append(Layer(nodes_per_layer[i]))  # nodes_per_layer does not include nodes of input and output
    layers.append(Layer(outputs))
    return layers


def create_nodes(layers):
    layer_length = len(layers)
    for layerIdx in range(1, layer_length):
        layer = layers[layerIdx]
        for i in range(0, layer.num_nodes):
            layer.append_node(Neuron(layers[layerIdx - 1].get_num_nodes()))


def create_network(inputs, outputs, nodes_per_layer):
    layers = create_layers(inputs, outputs, nodes_per_layer)
    create_nodes(layers)
    return NeuralNetwork(layers)
    # network.feedforward_network(np.array([-3.0, -3.0]))
    # data = np.array([-5.2, 3.1, 3.2, 6])
    # true = np.array([.76, .311, .122])
    # learn_rate = 0.1
    # cycles = 1000
    # network.backprop_network(data, true, learn_rate, cycles)
    # print(network.feedforward_network(data))

    # Define dataset
    # data = np.array([
    #     [-2, -1],  # Alice
    #     [25, 6],  # Bob
    #     [17, 4],  # Charlie
    #     [-15, -6],  # Diana
    # ])
    # all_y_trues = np.array([
    #     np.array([1, 1]),  # Alice
    #     np.array([0, 0]),  # Bob
    #     np.array([0, 0]),  # Charlie
    #     np.array([1, 1]),  # Diana
    # ])
    #
    # emily = np.array([-7, -3])  # 128 pounds, 63 inches
    # frank = np.array([20, 2])  # 155 pounds, 68 inches
    #
    # print("Emily:", network.feedforward_network(emily))  # 0.951 - F
    # print("Frank:", network.feedforward_network(frank))  # 0.039 - M
    # network.train(data, all_y_trues, 25000)
    #
    # print("Emily:", network.feedforward_network(emily))  # 0.951 - F
    # print("Frank:", network.feedforward_network(frank))  # 0.039 - M

    # network(inputs=3, outputs=1, nodes_per_layer=(2,), activation="sigmoid")
    # network(inputs=2, outputs=2, nodes_per_layer=(2,))


def turn_node_to_string(layers):
    master_string = ''
    num_layers = len(layers)
    class_list = []
    output_list = []
    layer_names = []
    for layerIdx in range(0, num_layers - 1):
        layer = layers[layerIdx]
        this_list = []
        for i in range(0, layer.num_nodes):
            master_string += '~~'
            name_of_node = "self.L" + str(layerIdx) + "n" + str(i)
            this_list.append(name_of_node)
            name_of_node += " = Neuron(" + "number_of_weights=" + str(layers[layerIdx - 1].num_nodes) + ""
            name_of_node += ")"
            master_string += name_of_node + '\n'
        class_list.append(this_list)
        master_string += '\n'
    for i in range(0, layers[-1].num_nodes):
        master_string += '~~'
        name_of_node = "self.On" + str(i)
        output_list.append(name_of_node)
        name_of_node += " = Neuron(" + "number_of_weights=" + str(layers[-2].num_nodes) + ")"
        master_string += name_of_node + '\n'
    master_string += '\n'
    for layerIdx in range(0, num_layers - 1):
        name = "layer" + str(layerIdx)
        layer_names.append(name)
        master_string += "~~" + name + " = Layer(num_nodes=" + str(layers[layerIdx].num_nodes) + ")\n"
        master_string += "~~" + name + ".nodes = ["
        for j in range(0, len(class_list[layerIdx]) - 1):
            master_string += class_list[layerIdx][j] + ", "
        master_string += class_list[layerIdx][-1] + "]\n\n"
    master_string += "~~output = Layer(num_nodes=" + str(layers[-1].num_nodes) + ")\n"
    master_string += "~~output.nodes = ["
    master_string += ", ".join(output_list) + ']\n'
    master_string += "\n~~self.layers = [" + ", ".join(layer_names+['output']) + "]"
    return master_string
