from django.shortcuts import render


def home_page(request):
    context = {
        'title': 'Home Page',
        'content': 'This is the homepage for the Education Page'
    }
    return render(request, 'educate/home_page.html', context)
