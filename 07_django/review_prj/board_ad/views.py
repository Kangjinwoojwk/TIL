from django.shortcuts import render, redirect, get_object_or_404, get_list_or_404
from django.views.decorators.http import require_http_methods
from .models import Posting, Comment


# Create
@require_http_methods(['GET', 'POST'])  # 많은 다른 접근이 있다.제대로 한다면 else 아닌 elif 너무 길어져
def create_posting(request):
    if request.method == 'POST':
        posting = Posting()
        posting.title = request.POST.get('title')
        posting.content = request.POST.get('content')
        posting.save()
        return redirect('board_ad:posting_detail', posting_id=posting.id)
    else:
        return render(request, 'board_ad/new.html')


# Read
@require_http_methods(['GET'])
def posting_list(request):
    postings = Posting.objects.all()  # 빈 리스트라도 있어야 글을 쓰지
    # postings = get_list_or_404(Posting)
    return render(request, 'board_ad/list.html', {
        'postings': postings,
    })


@require_http_methods(['GET'])
def posting_detail(request, posting_id):
    posting = get_object_or_404(Posting, id=posting_id)
    comments = posting.comment_set.all()
    return render(request, 'board_ad/detail.html', {
        'posting': posting,
        'comments': comments,
    })


# Update
@require_http_methods(['GET', 'POST'])
def update_posting(request, posting_id):
    posting = get_object_or_404(Posting, id=posting_id)
    if request.method == 'POST':
        posting.title = request.POST.get('title')
        posting.content = request.POST.get('content')
        posting.save()
        return redirect('board_ad:posting_detail', posting_id=posting.id)
    else:
        return render(request, 'board_ad/edit.html',{
            'posting': posting
        })


@require_http_methods(['POST']) # POST요청에 대해서만 실행
def delete_posting(request, posting_id):
    posting = get_object_or_404(Posting, id=posting_id)
    posting.delete()
    return redirect('board_ad:posting_list')


@require_http_methods(['POST'])
def create_comment(request, posting_id):
    posting = get_object_or_404(Posting, id=posting_id)
    comment = Comment()
    comment.content = request.POST.get('comment')
    comment.posting = posting
    comment.save()
    return redirect('board_ad:posting_detail', posting_id)


@require_http_methods(['POST'])
def delete_comment(request, posting_id, comment_id):
    posting = get_object_or_404(Posting, id=posting_id)
    comment = get_object_or_404(Comment, id=comment_id)
    comment.delete()
    return redirect('board_ad:posting_detail', posting_id)
