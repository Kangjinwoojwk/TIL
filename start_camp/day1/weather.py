#import requests
from darksky import forecast
multicampus=forecast('97c84fd3fd00fb3fa3dcd9726cebfacd',37.501596, 127.039638)
print(multicampus['currently']['summary'])
print(multicampus['currently']['temperature'])
# url='https://api.darksky.net/forecast/97c84fd3fd00fb3fa3dcd9726cebfacd/37.501562,127.039660'

# res=requests.get(url)
# data=res.json()

# print(data['currently']['summary'])