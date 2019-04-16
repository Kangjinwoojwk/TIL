from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.http import require_GET,require_POST, require_http_methods
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import login as auth_login, logout as auth_logout

# Create your views here.
@require_http_methods(['GET', 'POST'])
def signup(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            return redirect('posts:post_list')
    else:
        form = UserCreationForm()
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
            form = AuthenticationForm(request, data=request.POST)
            if form.is_valid():
                # DO LOGIN
                auth_login(request, form.get_user())
                return redirect(request.GET.get('next') or 'posts:post_list')
        # 사용자가 로그인 화면을 요청할때
        else:
            form = AuthenticationForm()
        return render(request, 'accounts/login.html',{
            'form': form,
        })


def logout(request):
    auth_logout(request)
    return redirect('posts:post_list')