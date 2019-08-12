function sleep_3s() {
    setTimeout(()=>{
        console.log('Wake up!')
    }, 3000)
}

const logEND = () =>{
    console.log('END')
};
console.log('Start sleeping');
sleep_3s();
console.log('End of Program');