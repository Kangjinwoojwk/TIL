import requests
import datetime
import os

KEY = os.getenv('KOBIS_KEY')
DOMAIN = f'http://www.kobis.or.kr/kobisopenapi/webservice/rest/movie/searchMovieList.json?key={KEY}'
date = datetime.datetime.now()
date -= datetime.timedelta(days=1)
date = date.strftime('%Y%m%d').replace(' ', '')
print(date)

# for year in range(500):
# while date.strftime('%Y%m%d') > '20110101':
#     date -= datetime.timedelta(days=1)
#     print(date.strftime('%Y %m %d').replace(' ', ''))
    # curPage = 1
    # naver_uri = 'https://openapi.naver.com/v1/search/movie.json?query='
    # headers = {
    #     'X-Naver-Client-Id': os.getenv('NAVER_CLIENT_ID'),
    #     'X-Naver-Client-Secret': os.getenv('NAVER_CLIENT_SECRET')
    # }
    # data = requests.get(naver_uri + '1999020', headers=headers).json()
    # print(data)

    # while True:
    #     URL = DOMAIN + f'&openStartDt={year}&openEndDt={year}&itemPerPage=100&curPage={curPage}'
    #     data = requests.get(URL).json()['movieListResult']['movieList']
    #     for i in data:
    #         if 1:
    #             pass
    #         print(i)
    #     break