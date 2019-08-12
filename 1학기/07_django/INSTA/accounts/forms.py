from django.contrib.auth.forms import UserCreationForm, UserChangeForm, AuthenticationForm
# 회원 가입, 정보수정, 로그인에 관련된 form을 가지 것들 import
from .models import User
class CustomUserCreationForm(UserCreationForm):

    class Meta(UserCreationForm):
        model = User
        fields = ('username', 'email', 'first_name', 'last_name')

class CustomUserAuthenticationsForm(AuthenticationForm):

    class Meta(AuthenticationForm):
        model = User

# class CustomUserChangeForm(UserCreationForm)