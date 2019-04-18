from django.db import models
from faker import Faker
faker = Faker()
# Create your models here.

# class User(models.Model):
#     name = models.CharField(max_length=10)
#
# class Profie(models.Model):
#     age = models.IntegerField()
#     address = models.CharField(max_length=200)
#     user = models.OneToOneField(User, on_delete=models.CASCADE)


class Client(models.Model):
    name = models.CharField(max_length=30)
    class Meta:
        ordering = ('name', )

    @classmethod
    def dummy(cls, n):
        for i in range(n):
            cls.objects.create(name=faker.name())


class Hotel(models.Model):
    name = models.CharField(max_length=30)
    clients = models.ManyToManyField(Client, related_name='hotels')  # 알아서M:N릴레이션 만들어 준다

    @classmethod
    def dummy(cls, n):
        for i in range(n):
            cls.objects.create(name=faker.company())


class Student(models.Model):
    name = models.CharField(max_length=30)


class Lecture(models.Model):
    title = models.CharField(max_length=100)


class Enrolment(models.Model):
    student = models.ForeignKey(Student, on_delete=models.PROTECT)
    lecture = models.ForeignKey(Lecture, on_delete=models.PROTECT)