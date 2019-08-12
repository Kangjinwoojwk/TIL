from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.views.decorators.http import require_GET, require_POST, require_http_methods
# from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .forms import CustomUserAuthenticationsForm, CustomUserCreationForm
from django.contrib.auth import login as auth_login, logout as auth_logout
# from django.contrib.auth import get_user_model
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import User
from posts.forms import CommentModelForm
# Create your views here.
@require_http_methods(['GET', 'POST'])
def signup(request):
    if request.method == 'POST':
        form = CustomUserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect('posts:post_list')
    else:
        form = CustomUserCreationForm()
    return render(request, 'accounts/signup.html', {
        'form': form,
    })

@require_http_methods(['GET', 'POST'])
def login(request):
    # 우선, 사용자가 로드인 되어 있는지
    if request.user.is_authenticated:
        return redirect('posts:post_list')
    else: # 로그인 안되어 있다면
        # 사용자가 로그인 데이터를 넘겼을 때
        if request.method == 'POST':
            form = CustomUserAuthenticationsForm(request, data=request.POST)
            if form.is_valid():
                # DO LOGIN
                user = form.get_user()
                auth_login(request, form.get_user())
                messages.add_message(request, messages.SUCCESS, f'welcome back,{user.username}')
                messages.add_message(request, messages.INFO, f'마지막 로그인은 {user.last_login}입니다.')
                print(request.META.get('HTTP_REFERER', '/insta/'))
                return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/insta/'))
        # 사용자가 로그인 화면을 요청할때
        else:
            form = CustomUserAuthenticationsForm()
        return render(request, 'accounts/login.html',{
            'form': form,
        })


def logout(request):
    auth_logout(request)
    messages.add_message(request, messages.SUCCESS, f'Logout Successfully')
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/insta/'))


def user_detail(request, username):
    user_info = User.objects.get(username=username)
    return render(request, 'accounts/user_detail.html', {
        'user_info': user_info,
        'comment_form': CommentModelForm()
    })


@login_required
@require_POST
def toggle_follow(request, username):
    sender = request.user
    receiver = get_object_or_404(User, username=username)
    if sender != receiver:
        if receiver in sender.followings.all():
            sender.followings.remove(receiver)
        else:
            sender.followings.add(receiver)
    return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/insta/'))
