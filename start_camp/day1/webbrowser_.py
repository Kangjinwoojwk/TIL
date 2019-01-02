import webbrowser

keywords=[
    '삼성전자 주가',
    '스마트 팩토리',
    '나가사와 마리나',
    '애로우 시즌7',
    'the 100 s6'
]
for keyword in keywords:
    url='https://www.google.com/search?q='+keyword
    webbrowser.open_new(url)

