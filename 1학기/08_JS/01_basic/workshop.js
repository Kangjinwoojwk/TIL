const concat = (str1, str2) => `${str1} - ${str2}`;
const check_long_str = string => string.length > 10;

if (check_long_str(concat('Happy', 'Hacking'))){
    console.log('LONG STRING')
}
else{
    console.log('SHORT STRING')
}
check_long_str(concat('Happy', 'Hacking'))? console.log('LONG STRING'):console.log('SHORT STRING');
