from django import forms
from .models import Post, Image, comment

class PostModelForm(forms.ModelForm): # 두가지 일, 1. 던지면 알아서 분류, 2. 화면 알아서 구성
    class Meta:
        model = Post
        fields = ['content', ]


class ImageModelForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['file', ]
        widgets ={
            'file': forms.FileInput(attrs={'multiple': True})
        }


class CommentModelForm(forms.ModelForm):
    class Meta:
        model = comment
        fields = ['content', ]