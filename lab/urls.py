from django.urls import path

from . import views

app_name = 'educate'

urlpatterns = [
    path('', views.home_page, name='home_page'),
]