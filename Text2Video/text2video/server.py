import os
import random
import time
from flask import Flask, request
import json

app = Flask(__name__)



@app.route('/text2video', methods=['POST'])
def post_service():
    try:
        data = request.get_json()
        if data and 'text' in data:
            current_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
            random_number = random.randint(1000, 9999)
            base_name = f"{current_time}_{random_number}"
            #video_path = f"/root/share/mp4s/{base_name}.mp4"
            video_path = f"/root/zhuxiaoxu/code/MuseTalk/results/avatars/avator_2/vid_output/{base_name}.mp4"
            link_path = f"http://localhost:8123/{base_name}.mp4"
            text_path = f"/root/share/text_server/{base_name}.txt"
            with open(text_path, 'w', encoding='utf-8') as f:
                f.write(data['text'])
            while True:
                if os.path.exists(video_path):
                    return json.dumps({"video_path": video_path, "link_path": link_path})
                time.sleep(1)
        return json.dumps({"error": "没有收到有效的JSON数据"}), 400
    except Exception as e:
        return json.dumps({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(port=8765, debug=True)

