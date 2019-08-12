const me = {
    name: 'KangJinwoo',
    'phone number':'010-1324-1324', //키에 띄어쓰기가 있으면''안에 쓴다.
    status:{
        mental: 'unnormal',
        body: 'almost dead',
    }
};

me.name; // KangJinwoo
me['name']; // KangJinwoo
me['phone number']; // 010-1324-1324
me.status; // {mental:'unnormal', body:'almost dead'}
me.status.mental; // unnormal