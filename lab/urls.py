from django.urls import path

from lab import views

app_name = 'educate'

urlpatterns = [
    path('', views.home_page, name='home_page'),
    path(r'loading/', views.loading, name='loading_page'),
    path(r'main/', views.main_page, name='main_page'),
    path(r'post_script/', views.post_script, name='post_script'),
    path(r'activations/', views.activations, name='activations'),
]