const DOMAIN = 'https://jsonplaceholder.typicode.com/';
const RESOURCE = 'posts/';
const QUERY_STRING = '';
// req 대리인 XHR 객체 생성
const URL = DOMAIN+RESOURCE+QUERY_STRING;
const XHR = new XMLHttpRequest();

// XHR 요청 발사 준비 (method, url)
// 요청 만들기 -> 정보 담기-> 보내기-> 기다리기->처리
XHR.open('POST', URL);
// 어떤 정보인지 인식 시켜 줘야한다. 없으면 원래 못 받는다.
XHR.setRequestHeader(
    'Content-Type',
    'application/json;charset=UTF-8'
);
// XHR 요청 발사!
XHR.send(
    JSON.stringify({"title": "NewPost", "body": "This is New Post", "userId": 1 })
);

XHR.addEventListener('load', e =>{
   const result = JSON.parse(e.target.response);
   console.log(result);
});