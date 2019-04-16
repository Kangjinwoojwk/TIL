from django.shortcuts import render, redirect
from .models import Writer, Book, Chapter
from .forms import *

# Create your views here.

def create(request):
    if request.method == 'POST':
        form = WriterModelFrom(request.POST)
        if form.is_valid():
            form.save()
            return redirect('성공!')
        else:  # 실패하면
            return redirect('실패!')

        # writer = Writer()
        # writer.name = request.POST.get('name')
        # writer.save()
        # return redirect()
    elif request.method == 'GET':
        form = WriterModelFrom()
    return render(request, 'new.html', {
        'form': form,
    })


def update(request, id):
    writer = Writer.objects.get(id=id)
    if request.method == 'POST':
        form = WriterModelFrom(request.POST)
        if form.is_valid():
            form.save()
            return redirect('detail')
        else:
            pass
        # writer.name = request.POST.get('name')

    elif request.method == 'GET':
        form = WriterModelFrom(instance=writer)
    return render(request, 'edit.html', {
        'form': form,
    })