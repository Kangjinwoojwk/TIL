// function addAll(numbers=[]){
//     let sum = 0;
//     numbers.forEach(number => sum += number);
//     return sum;
// }
function SubAll(){}
function mulAll(){}

module.exports = { // 키, 밸류가 같은 때만 사용가능
    addAll(numbers=[]) {
        let sum = 0;
        numbers.forEach(number => sum += number);
        return sum;
    },
    subAll(){
        let sum = 0;
        numbers.forEach(number => sum -= number);
        return sum;
    },
    mulAll(){
        let sum = 1;
        numbers.forEach(number => sum *= number);
        return sum;
    },
    name:'kang'
};
phoneNumber = '01012341234';
module.exports.phoneNumber=phoneNumber;
