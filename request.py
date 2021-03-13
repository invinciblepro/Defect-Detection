import requests

url = 'http://localhost:5000/results'
r = requests.post(url,json={'media': open('test.jpg', 'rb')})
print(r.json())