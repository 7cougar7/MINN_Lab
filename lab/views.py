from urllib.parse import urlencode

from django.shortcuts import render, redirect
from django.urls import reverse


def home_page(request):
    context = {
        'title': 'Home Page',
        'content': 'This is the homepage for the Labs Page'
    }
    return render(request, 'lab/home_page.html', context)


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
