from django.shortcuts import render
import os


def home_page_view(request, *args, **kwargs):
    path = os.path.join('home_page.html')
    return render(request, path, {})
