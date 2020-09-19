from urllib.parse import urlencode

from django.http import JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
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


def post_function_type(request):
    functions = {
        '0': 'def sigmoid(x):\n~return 1 / (1 + np.exp(-x))\n\ndef sigmoid_prime(x):\n~tfx = sigmoid(x)\n~return fx * (1 - fx)\n',
        '1': 'def tanh(x):\n~return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))\ndef tanh_prime(x):\n~t = tanh(x)\n~return 1 - t ** 2\n',
        '2': 'def relu(x):\n~return max(0, x)\n\ndef relu_prime(x):\n~return 1 if x > 0 else 0\n',
        '3': 'def parametric_relu(x, alpha=0.01):\n\n~return max(alpha * x, x)\ndef parametric_relu_prime(x, alpha=0.01):\n~return 1 if x > 0 else alpha\n',
        '4': 'def elu(x, alpha):\n~return x if x >= 0 else alpha * (np.exp(x) - 1)\n\ndef elu_prime(x, alpha):\n~return 1 if x > 0 else alpha * np.exp(x)\n',
        '5': 'def swish(x):\n~return x / (1 + np.exp(-x))\n\ndef swish_prime(x):\n~sig = sigmoid(x)\n~swi = swish(x)\n~return swi + (sig * (1 - swi))\n',
        '6': 'def linear(x, slope):\n~return x * slope\n\ndef linear_prime(slope):\n~return slope\n',
        '7': 'def binary(x):\n~return 1 if x > 0 else 0\n\ndef binary_prime():\n~return 0\n'
    }
    response = {
        'functionCode': format_string(functions.get(request.POST.get('funcVal', '0')))
    }
    return JsonResponse(response)
