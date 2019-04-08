from django.shortcuts import render
from .models import Article
# Create your views here.

def article_new(request):
    pass

def article_create(request):
    pass

def article_detail(request):
    pass

def article_list(request):
    pass

def article_read(request, id):
    pass

def article_update(request):
    pass

def article_edit(request):
    pass

def article_delete(request):
    pass

def index(request):
    return render(request, 'board/index.html')


def greeting(request, name, role):
    if name == 'admin':
        return render(request, 'board/greeting.html', {
            'role': 'MASTER_USER',
            'name': 'MASTER',
        })
    else:
        return render(request, 'board/greeting.html', {
            'name': name,
            'role': role,
        })