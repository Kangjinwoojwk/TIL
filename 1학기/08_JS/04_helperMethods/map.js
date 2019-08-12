// ES5 for loop
// var numbers = [1,2,3];
// var doubleNumbers = [];
//
// for(var i = 0; i < numbers.length;i++){
//     doubleNumbers.push(numbers[i] * 2);
// }
//
// console.log(doubleNumbers);
// ES6+
const numbers = [1, 2, 3];
function double(n) {
    return n * 2;
}

const doubleNumbers = numbers.map(double);
const tripleNumbers = numbers.map(number => {
    return number * 3;
});
console.log(tripleNumbers);

const image = [
    { height : 34, width:39},
    { height : 54, width:19},
    { height : 83, width:75},
];
const imageAreas = image.map(image =>{
    return image.height * image.width;
});
console.log(imageAreas);

/*
    아래의 pluck 함수를 완성하세요.
    pluck은 함수는 배열(array)과 요소 이름의 문자열을
 */
function pluck(array, property) {
    return array.map(thing =>{
        if (thing[property]) {
            return thing[property];
        }
    });
}
const paints = [
    {color : 'red'},
    {color : 'blue'},
    {color : 'white'},
    {smell : 'ughh'}
];
console.log(pluck(paints, 'color'));  // ['red', 'blue', 'white']
pluck(paints, 'smell'); // ['ughh']