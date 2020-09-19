from urllib.parse import urlencode

from django.shortcuts import render, redirect
from django.urls import reverse


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
