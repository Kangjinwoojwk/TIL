from django.db import models
from django_extensions.db.models import TimeStampedModel
from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill
import os
ENV = os.environ.get('ENVIRONMENT', 'development')
if ENV == 'development':  # 개발자와 구분
    from faker import Faker
faker = Faker()


# Create your models here.
class Post(TimeStampedModel):
    content = models.CharField(max_length=140)
    # 이미지를 편집해서 저장할 것이다.
    # image = ProcessedImageField(
    #     blank=True,
    #     upload_to='posts/images',
    #     processors=[ResizeToFill(600, 600)],
    #     format='JPEG',
    #     options={'quality': 90},
    # )

    # created_at = models.DateTimeField(auto_now_add=True)
    # updated_at = models.DateTimeField(auto_now=True)
    @classmethod
    def dummy(clscls, n):
        for _ in range(n):
            Post.objects.create(content=Faker.text(120))


class Image(TimeStampedModel):
    post = models.ForeignKey(Post, on_delete=models.CASCADE)
    file = ProcessedImageField(
        blank=True,
        upload_to='posts/images',
        processors=[ResizeToFill(600, 600)],
        format='JPEG',
        options={'quality': 90},
    )