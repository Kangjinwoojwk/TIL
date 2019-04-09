from django.urls import path
from . import views

urlpatterns = [
    path('index/', views.index), # Domain/board/index
    path('greeting/<str:name>/<str:role>/', views.greeting), # Domain/board/greeting /john
    # Create
    # /articles/new => html
    path('articles/new/', views.article_new),
    # /articles/create => DB new record
    path('articles/create/', views.article_create),

    # Read
    # /articles=> html(all articles)
    path('articles/', views.article_list),
    # /articles => html(article with id 1)
    path('articles/<int:id>/', views.article_detail),

    # Updata
    # /articles/1/edit => html(article id = 1 수정하는 화면)
    path('articles/<int:id>/edit/', views.article_edit),
    # /articles /1/update => DB update article id = 1
    path('articles/<int:id>/update/', views.article_update),

    # Delete
    # /articels/1/delete => delete article ids = 1
    path('articles/<int:id>/delete/', views.article_delete),
]