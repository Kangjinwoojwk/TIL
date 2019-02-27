# 2월 월말 평가 Web

## 1. HTML/CSS

* 예시 결과 보고 CSS 작성, HTML에 적용

태그 선택자, 클래스 선택자, id선택자

```css
p {
    width = 100%;
}
.class {
    width = 50px;
}
#id {
    width = 50rem;
}
```

HTML에 어떻게 적용하는가? 부트스트랩은 밑과 같다.

```html
<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css" integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T" crossorigin="anonymous">
<link rel="stylesheet" href="{% static 'simple_board/css/bootstrap.css' %}" type="text/css"/>
```

href에 넣어주는게 바껴야한다

## 2. Bootstrap

* Grid System

```html
 <div class="float-left card position-fixed col-4 col-sm-4 col-md-4 col-lg-3 col-xl-3">
```

* 예시보고 알맞는 클래스 채워 넣기, 사이트 접속 불가능, 반응형 과 breakpoint는 문제에 주어짐
* 공식 문서 반드시 볼 것, offsetting columns까지, 세로 정렬 출제 x

## 3. Django

* R,D 중 하나 작성
* 프로젝트 코드 제공, 서버 실행 불가능
* `views.py`에 새로운 함수 작성하여 페이지 만드는 법, template(html) 파일 만드는 법 숙지 필요.
* Django Template Language의 기본적인 사용법, '반복문', '조건문', '템플릿 상속(extends)', '페이지 출력(render)' 숙지 필요.
* 부분 점수를 위해서 최대한 많이 작성할 것.

```html
{% extends 'simple_board/base.html' %}
{% load static %}
{% block body %}

<h1>All articles</h1>
<header>
    <img src = "{% static 'simple_board/images/header.png' %}" alt="header">
</header>
{% if articles %}
<ul>
    {% for article in articles %}
        <li>
            <a href="{% url 'simple_board:article_detail' article.id %}">{{ article.title }}</a>
        </li>
    {% endfor %}
</ul>
<a href="{% url 'simple_board:article_create' %}">글쓰기</a>

{% endif %}
{% endblock %}
```

```python
from django.urls import path
from . import views

app_name = 'simple_board'

urlpatterns = [
    # URL about articles
    path('', views.article_list, name = 'article_list'),
    path('create/', views.article_create, name = 'article_create'),
    # /simple_board/1/
    path('<int:article_id>/', views.article_detail, name = 'article_detail'),
    path('<int:article_id>/update/', views.article_update, name = 'article_update'),
    path('<int:article_id>/delete/', views.article_delete, name = 'article_delete'),
    path('<int:article_id>/comments/create/',views.comment_create, name='comment_create'),
    path('<int:article_id>/comments/<int:comment_id>/delete/',views.comment_delete, name='comment_delete'),
]
```

```python
from django.shortcuts import render, get_object_or_404, redirect
from django.http import HttpResponseRedirect
from django.urls import reverse

from .models import Article, Comment
# Create your views here.
"""
actions about model article
"""
def article_list(request):
    articles = Article.objects.all()
    return render(request, 'simple_board/list.html',{
        'articles':articles,
    })
    
def article_detail(request, article_id):
    article = get_object_or_404(Article, id = article_id)
    comments = article.comment_set.all()
    return render(request, 'simple_board/detail.html',{
        'article':article,
        'comments':comments,
    })
    
def article_create(request):
    if request.method == 'GET':
        return render(request, 'simple_board/new.html')
    else:
        article = Article()
        article.title = request.POST.get('title')
        article.content = request.POST.get('content')
        article.save()
    return redirect('simple_board:article_detail', article.id)
    
def article_update(request, article_id):
    article = get_object_or_404(Article, id = article_id)
    if request.method=='GET':
        return render(request, 'simple_board/edit.html',{'article':article})
    else:
        article.title=request.POST.get('title')
        article.content=request.POST.get('content')
        article.save()
    return redirect('simple_board:article_detail', article.id)

def article_delete(request, article_id):
    if request.method=='GET':
        return redirect('simple_board:article_detail',article_id)
    else:
        article = Article.objects.get(id = article_id)
        article.delete()
        return redirect('simple_board:article_list')
"""
actions about model Comment
"""
def comment_create(request, article_id):
    if request.method == 'POST':
        comment = Comment()
        comment.content = request.POST.get('content')
        comment.article = get_object_or_404(Article, id=article_id)
        comment.save()
    return redirect('simple_board:article_detail', article_id)
    
def comment_delete(request, article_id, comment_id):
    if request.method == "POST":
        comment = Comment.objects.get(id=comment_id)
        comment.delete()
    return redirect('simple_board:article_detail', article_id)
```

