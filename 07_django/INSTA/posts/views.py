from django.shortcuts import render, redirect, get_object_or_404, HttpResponseRedirect
from django.views.decorators.http import require_GET, require_POST, require_http_methods
from django.contrib.auth.decorators import login_required
from .models import Post, Image
from .forms import PostModelForm, ImageModelForm, CommentModelForm

@login_required
@require_http_methods(['GET', 'POST'])
def create_post(request):
    # POST 방식으로 넘온 Data 를 ModelForm 에 넣는다.
    if request.method == 'POST':
        # POST 방식으로 넘온 Data 를 ModelForm 에 넣는다.
        post_form = PostModelForm(request.POST)
        # Data 검증을 한다.
        if post_form.is_valid():

            # 통과하면 저장한다.
            post = post_form.save(commit=False)
            post.user = request.user
            post.save()
            for image in request.FILES.getlist('file'):
                request.FILES['file'] = image
                # files 명시 하지 않으면 맨 앞의 data가 다시 들어간다.
                image_form = ImageModelForm(files=request.FILES)
                if image_form.is_valid():
                    # image = Image()
                    # image.file = request.FILE.get('srasds')
                    image = image_form.save(commit=False)# 저장은 아직인 상태
                    image.post = post
                    image.save()

            return redirect('posts:post_list')
        else:
            # 실패하면, 다시 data 입력 form 을 준다.
            pass
    # GET 방식으로 요청이 오면,
    else:
        # 새로운 Post 를 쓸 form을 만든다.
        post_form = PostModelForm()
    image_form = ImageModelForm()
    return render(request, 'posts/form.html', {
        'post_form': post_form,
        'image_form': image_form,
    })


@require_http_methods(['GET', 'POST'])
def update_post(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if post.user == request.user: # 지금 수정자가 post 원작성자인가?
        if request.method == 'POST':
            post_form = PostModelForm(request.POST, instance=post)
            if post_form.is_valid():
                post_form.save()
                return redirect('posts:post_list')
        else:
            post_form = PostModelForm(instance=post)
    else: # 작성자와 요청 보낸 user가 다르다면
        # 401 forbiden 금지됨
        return redirect('posts:post_list')
    return render(request, 'posts/form.html',{
        'post_form': post_form,
    })


@require_GET
def post_list(request):
    if request.GET.get('next'):
        return redirect(request.GET.get('next'))
    posts = Post.objects.all()
    comment_form = CommentModelForm()
    return render(request, 'posts/list.html', {
        'posts': posts,
        'comment_form': comment_form,
    })


@login_required
@require_POST
def create_comment(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    comment_form = CommentModelForm(request.POST)
    if comment_form.is_valid():
        comment = comment_form.save(commit=False)
        comment.user = request.user
        comment.post = post
        comment.save()
        return HttpResponseRedirect(request.META.get('HTTP_REFERER'))
    # TODO: else: => if comment is not valid than?

# @login_required
# def create_like(request, post_id):
#     user = request.user
#     post = get_object_or_404(Post, id=post_id)
#     post.likey_users.add(user)

@login_required
@require_POST
def togle_likey(request, post_id):
    user = request.user
    post = get_object_or_404(Post, id=post_id)
    # if post.likey_users.filter(id=user.id).exists():  # 찾으면 [value] 없으면[]
    if user in post.likey_users.all():
        post.likey_users.remove(user)
    else:
        post.likey_users.add(user)
    return redirect('posts:post_list')