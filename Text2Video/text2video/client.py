import requests
import json

url="http://127.0.0.1:8765/text2video"
data={"text":"在一个宁静的小村庄里，住着一位年迈的画家。"}

headers={"Content-Type":"application/json"}

response=requests.post(url,data=json.dumps(data),headers=headers)
print(response.json())
