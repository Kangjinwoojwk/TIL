// 하나의 함수로 안되나?
// 숫자로 이루어진 배열의 요소들을 각각 [???] 한다. [???] 는 알아서 해라.
const numberEach=(numbers, callback) =>{
    let acc;
    for (const number of numbers){
        acc = callback(number, acc);
    }
    return acc;
};
const adder = (number, sum = 0) =>{
    return sum + number;
};
console.log(numberEach([1, 2, 3, 4, 5], (number, sum = 0)=>sum + number));
console.log(numberEach([1, 2, 3, 4, 5], (number, acc = 1)=>acc * number));



// 인자로 배열을 받는다. 해당 배열의 모든 요소를 더한 숫자를 return
const numbersEachAdd = numbers =>{
    let acc = 0;
    for (const number of numbers){
        acc += number;
    }
    return acc;
};
// 인자로 배열을 받는다. 해당 배열의 모든 요소를 뺀 숫자를 return
const numbersEachSub = numbers =>{
    let acc = 0;
    for (const number of numbers){
        acc -= number;
    }
    return acc;
};
// 인자로 배열을 받는다. 해당 배열의 모든 요소를 곱한 숫자를 return
const numbersEachMul = numbers =>{
    let acc = 1;
    for (const number of numbers){
        acc *= number;
    }
    return acc;
};
// console.log(numbersEachAdd([1, 2, 3, 4, 5]));
// console.log(numbersEachSub([1, 2, 3, 4, 5]));
// console.log(numbersEachMul([1, 2, 3, 4, 5]));