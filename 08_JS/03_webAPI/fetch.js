// ES6+ 부터 등장

const DOMAIN = 'https://jsonplaceholder.typicode.com/';
const RESOURCE = 'posts/';
const QUERY_STRING = '';
const URL = DOMAIN+RESOURCE+QUERY_STRING;
// (만들기) -> 정보 담기 -> 보내기 -> 기다리기 -> 처리

const getRequest = URL => {
    fetch(URL)// 만들기 -> 정보 담기 -> 보내기
        .then(response=>response.json()) // 기다리기-> 파싱함
        .then(parseData=>console.log(parseData)); // 파싱한 데이터 출력
};
const postRequest = URL =>{
    fetch(URL, {
        method: 'POST',
        body: JSON.stringify({
            title:'new post',
            content:'new content',
            userId:1
        }),
        headers: {
            'Content-type': 'application/json; charset=UTF-8'
        }

    }).then(response=>response.json()) // 기다리기-> 파싱함
      .then(parseData=>console.log(parseData)); // 파싱한 데이터 출력
};
postRequest(URL);