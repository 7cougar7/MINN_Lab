from django.urls import path

from lab import views

app_name = 'educate'

urlpatterns = [
    path('', views.home_page, name='home_page'),
    path(r'loading/', views.loading, name='loading_page'),
    path(r'main/', views.main_page, name='main_page'),
    path(r'backend_call/', views.backend_call, name='backend_call'),
    path(r'backend_call_func/', views.backend_request_function, name='backend_request_function'),
]