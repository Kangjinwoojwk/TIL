from django.contrib import admin
from .models import Posting, Comment

class PostingModelAdmin(admin.ModelAdmin): # 두가지를 admin페이지에서 보기 위한 객체
    readonly_fields = ('created_at', 'updated_at') # readonly로 넘기면 변경 불가
    list_display = ('id', 'title', 'created_at', 'updated_at') # 글 리스트 화면 변경
    list_display_links = ('id', 'title') # 링크 되는 애들 설정

admin.site.register(Posting, PostingModelAdmin) # 같이 넘겨줘야 의미가진다

class CommentModelAdmin(admin.ModelAdmin):
    readonly_fields = ('created_at', 'updated_at')  # readonly로 넘기면 변경 불가
    list_display = ('id', 'posting', 'content', 'created_at', 'updated_at')  # 글 리스트 화면 변경
    list_display_links = ('id', 'content')  # 링크 되는 애들 설정

admin.site.register(Comment, CommentModelAdmin)