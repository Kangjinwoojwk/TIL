from django.db import models
from django_extensions.db.models import TimeStampedModel, TitleDescriptionModel
# Create your models here.


class MagazineArticle(TitleDescriptionModel, TimeStampedModel):
    content = models.TextField()


class TimeStamp(models.Model):
    created_at = models.DateTimeField(auto_now_add=True)
    modified = models.DateTimeField(auto_now=True)
    class Meta: # 이게 없으면 TimeStamp라는 테이블이 따로 생긴다. 의도와 다르다.
        abstract = True


class Writer(TimeStamp):
    name = models.CharField(max_length=100, default='')


class Book(TimeStamp):
    author = models.ForeignKey(Writer, on_delete=models.PROTECT)
    title = models.CharField(max_length=200, default='', unique=True)
    description = models.TextField()


class Chapter(TitleDescriptionModel, TimeStampedModel):
    # title description, created, modified
    book = models.ForeignKey(Book, on_delete=models.CASCADE)


class Student(models.Model):
    name = models.CharField(max_length=20)


class Message(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    content = models.CharField(max_length=10)


class Reply(models.Model):
    content = models.CharField(max_length=30)
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    message = models.ForeignKey(Message, on_delete=models.CASCADE)












