const numbers = [1, 2, 3, 4];
numbers[0]; // 1
numbers[-1]; //undefined
numbers[4]; //undefined
numbers.length; // 4
/* 원본이 달라지는 methods*/
numbers.reverse(); // [4, 3, 2, 1]
numbers;//[4, 3, 2, 1]
numbers.reverse(); // [1, 2, 3, 4]
numbers;//[1, 2, 3, 4]
numbers.push('a'); // 5 new lenth
numbers; // [1, 2, 3, 4, 'a']
numbers.pop(); // 'a'
numbers; // [1, 2, 3, 4]
numbers.unshift('a'); // 5 new lenth
numbers; // ['a', 1, 2, 3, 4]
numbers.shift(); // 'a'
numbers; // [1, 2, 3, 4]
/* Copy, 다른 결과 return , 원본 안바뀜*/
numbers.includes(1); // true
numbers.includes(0); // false

numbers.push('a', 'a'); // test
numbers.indexOf('a'); // 4 뒤에 중복 안잡고 끝
numbers.indexOf('b'); // -1 ->없음

numbers.join('-'); // 1-2-3-4-a-a
numbers.join(''); // 1234aa
numbers.join(); //1, 2, 3, 4, a, a

