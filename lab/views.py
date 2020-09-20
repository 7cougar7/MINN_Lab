from urllib.parse import urlencode

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from lab.NetworkTranslation import create_network, turn_node_to_string
import re

INDENTION = '    '


def format_string(string):
    return re.sub(r'~', INDENTION, string)


def home_page(request):
    context = {
        'title': 'Home Page',
        'content': 'This is the homepage for the Labs Page'
    }
    return render(request, 'lab/loading.html', context)


def main_page(request):
    context = {
        'title': 'MINN Lab',
        'activation_functions': {'Sigmoid': 0, 'Hyperbolic Tangent': 1, 'ReLU': 2, 'Leaky/Parametric ReLU': 3, 'ELU': 4,
                                 'Swish': 5, 'Linear': 6, 'Binary': 7}
    }
    return render(request, 'lab/main.html', context)


def loading(request):
    return render(request, 'lab/loading.html')


def backend_call(request):
    context = {
        'title': 'Backend Call',
        'result': request.GET.get('result', None)
    }
    return render(request, 'lab/backend_call.html', context)


def backend_request_function(request):
    if request.POST:
        num1 = int(request.POST.get('number-input'))
        num2 = int(request.POST.get('number-input1'))
        product = num1 * num2

        base_url = reverse('lab:backend_call')
        query_string = urlencode({'result': product})
        url = '{}?{}'.format(base_url, query_string)
        return redirect(url)
    return redirect(reverse('lab:backend_call'))


def function_type(func_val, alpha):
    int_func_val = int(func_val)
    int_alpha_val = int(alpha)
    function_inputs = ['self.sum_val']
    deriv_function_inputs = ['node.sum_val', 'input', 'node.weights[i]']

    functions = {
        '0': 'def sigmoid(x):\n~return 1 / (1 + np.exp(-x))\n\ndef sigmoid_prime(x):\n~fx = sigmoid(x)\n~return fx * (1 - fx)\n',
        '1': 'def tanh(x):\n~return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))\n\ndef tanh_prime(x):\n~t = tanh(x)\n~return 1 - t ** 2\n',
        '2': 'def relu(x):\n~return max(0, x)\n\ndef relu_prime(x):\n~return 1 if x > 0 else 0\n',
        '3': 'def parametric_relu(x):\n~return max(ALPHA * x, x)\n\ndef parametric_relu_prime(x):\n~return 1 if x > 0 else ALPHA\n',
        '4': 'def elu(x):\n~return x if x >= 0 else ALPHA * (np.exp(x) - 1)\n\ndef elu_prime(x):\n~return 1 if x > 0 else ALPHA * np.exp(x)\n',
        '5': 'def swish(x):\n~return x / (1 + np.exp(-x))\n\ndef swish_prime(x):\n~sig = sigmoid(x)\n~swi = swish(x)\n~return swi + (sig * (1 - swi))\n',
        '6': 'def linear(x):\n~return x * ALPHA\n\ndef linear_prime():\n~return ALPHA\n',
        '7': 'def binary(x):\n~return 1 if x > 0 else 0\n\ndef binary_prime():\n~return 0\n'
    }

    function_call = {
        '0': 'sigmoid(',
        '1': 'tanh(',
        '2': 'relu(',
        '3': 'parametric_relu(',
        '4': 'elu(',
        '5': 'swish(',
        '6': 'linear(',
        '7': 'binary('
    }

    deriv_function_call = {
        '0': 'sigmoid_prime(',
        '1': 'tanh_prime(',
        '2': 'relu_prime(',
        '3': 'parametric_relu_prime(',
        '4': 'elu_prime(',
        '5': 'swish_prime(',
        '6': 'linear_prime(',
        '7': 'binary_prime('
    }
    desired_function_call = function_call.get(func_val)
    desired_deriv_function_call = deriv_function_call.get(func_val)
    function_calls = []
    for func_input in function_inputs:
        temp_call = desired_function_call + func_input
        function_calls.append(temp_call)
    for func_input in deriv_function_inputs:
        temp_call = desired_deriv_function_call
        if int_func_val == 7:
            temp_call += ')'
        else:
            temp_call += func_input + ')'
        temp_call += ')'
        function_calls.append(temp_call)

    function_definition = ''
    if int_func_val == 3 or int_func_val == 4 or int_func_val == 6:
        function_definition += 'ALPHA = ' + alpha + '\n\n'
    function_definition += format_string(functions.get(func_val))
    functions = {
        'function_def': function_definition,
        'function_calls': function_calls,
    }
    return functions


def post_script(request):
    if request.POST:
        functions = function_type(request.POST.get('funcVal', '0'), request.POST.get('alphaVal', '7'))
        network = create_network(
            inputs=int(request.POST.get('inputVal', '1')),
            outputs=int(request.POST.get('outputVal', '1')),
            nodes_per_layer=[int(num_string) for num_string in request.POST.getlist('numNodesPerLayer[]', ['2'])]
        )
        nn_code = format_string('import numpy as np\n\n') + functions['function_def'] + format_string(r"""
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
        self.activation_func_val = 0

    def feedforward(self, inputs):
        # Weight inputs, add bias, and use activation function
        self.input_vec = inputs
        self.sum_val = np.dot(self.weights, inputs) + self.biases
        self.activation_func_val = """ + functions['function_calls'][0] + """
        return self.activation_func_val

    def get_weight(self, index):
        return self.weights[index]

    def set_weight(self, index, value):
        self.weights[index] = value

    def backprop_node(self, deriv):
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
            deriv = """ + functions['function_calls'][1] + """
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
    def __init__(self):
""" + format_string(turn_node_to_string(network.layers)) + """
        self.output_vec = np.array([])

    def feedforward_network(self, input_vec):
        self.inputs = input_vec
        output_vec = self.layers[1].feedforward_layer(input_vec)
        for layerIdx in range(2, len(self.layers)):
            output_vec = self.layers[layerIdx].feedforward_layer(output_vec)
        return output_vec

    def update_node_weights(self, node, node_number, layer_number, layer_in_front=0):
        inputs = node.input_vec
        if layer_number == len(self.layers) - 1:
            derivatives = np.array([])
            for input in inputs:
                deriv = """ + functions['function_calls'][2] + """
                derivatives = np.append(derivatives, deriv)
            delta_w = self.error[node_number] * inputs * derivatives
            node.weights += delta_w
        else:
            # dot product of derivatives of the output weights and the output errors
            scalars = np.array([])
            for i in range(0, self.layers[layer_number].num_nodes):
                weight_deriv = np.array([])
                for node in layer_in_front.nodes:
                    weight_deriv = np.append(weight_deriv, """ + functions['function_calls'][3] + """)
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
            for nodeIdx in range(0, current_layer.get_num_nodes()):
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
                self.backprop_network(data, true)""")
        return JsonResponse({'text': nn_code})
