from django import forms
from .models import *  # 해당에 있는거 전부 쓴다면 이렇게 하면 된다.

# Writer, Book, Chapter

class WriterModelFrom(forms.ModelForm):
    class Meta:
        model = Writer
        fields = '__all__'


class BookModelFrom(forms.ModelForm):
    class Meta:
        model = Book
        fields = '__all__'


class ChapterModelFrom(forms.ModelForm):
    class Meta:
        model = Chapter
        fields = '__all__'
