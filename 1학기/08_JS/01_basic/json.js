// ['a':'A'] json-JavaScriptObjectNotation 자바스크립트 물품 표현법
const myObject = {
    coffee:'No',
    iceCream:'Cookie and Cream',
};

const jsonData = JSON.stringify(myObject); // 문자열
console.log(typeof jsonData);

const parseData = JSON.parse(jsonData);
console.log(typeof parseData); //object