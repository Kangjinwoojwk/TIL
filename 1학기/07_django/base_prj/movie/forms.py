from django import forms
from .models import Movie


class MovieForm(forms.Form):
    title = forms.CharField(max_length=100)
    title_en = forms.CharField(max_length=100)
    audience = forms.IntegerField()
    open_date = forms.DateField(
        widget=forms.widgets.DateInput(attrs={'type': 'date'})  # 이게 없으면 날짜입력으로 나오지 않는다.
    )
    genre = forms.CharField(max_length=100)
    watch_grade = forms.CharField(max_length=100)
    score = forms.FloatField()
    poster_url = forms.CharField()  # text 필드는 없다. 대체한다
    description = forms.CharField(widget=forms.Textarea())


class MovieModelForm(forms.ModelForm):
    class Meta:
        model = Movie
        fields = '__all__'  # {'title', 'title_eng'}등 특정만 가져 올 수도 있다.
        widgets = {
            'open_date': forms.DateInput(attrs={'type': 'date'})
        }
