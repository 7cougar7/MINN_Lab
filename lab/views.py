from django.shortcuts import render


def home_page(request):
    context = {
        'title': 'Home Page',
        'content': 'This is the homepage for the Labs Page'
    }
    return render(request, 'lab/home_page.html', context)
