from django.db import models
from django_extensions.db.models import TimeStampedModel
from django.conf import settings
# Create your models here.
class Post(TimeStampedModel):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    title = models.CharField(max_length=50, default='')
    content = models.TextField()
    like_counts = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='likes',
        blank=True
    )
    def __str__(self):
        return self.title