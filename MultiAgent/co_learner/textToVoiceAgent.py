import requests
import json

def text2video(text):
    url = "http://127.0.0.1:8765/text2video"
    data = {"text": text}
    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(data), headers=headers)
    return response.json()

def main():
    text = "在一个宁静的小村庄里，住着一位年迈的画家。"
    response = text2video(text)
    print(response)

if __name__ == "__main__":
    main()