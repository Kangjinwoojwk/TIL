from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings
# Create your models here.


class User(AbstractUser):
    followings = models.ManyToManyField(
        # 생성전이라서 settings에서 불러 온 걸로 돌아가야한다.
        # 모델 아닌 스트링, 알아서 처리해줌 가져올때 get_user_model()로 해야 한다.
        settings.AUTH_USER_MODEL,
        related_name='followers',
        blank=True
    )

    def __str__(self):
        return self.username