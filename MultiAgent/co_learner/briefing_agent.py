import json
import os
import time
from datetime import datetime
from llm_chat import llm_chat_with_prompt

class BriefingAgent:
    def __init__(self, user_id):
        self.save_directory = os.path.join(os.getcwd(), "user_data", user_id, "briefings")
        self.briefing_list = []
        self.today_briefing = None
        # 判断目录是否存在，如果不存在则创建
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
        else:
            # 如果目录存在，则读取目录下的所有简报文件
            for filename in os.listdir(self.save_directory):
                if filename.endswith(".json"):
                    file_path = os.path.join(self.save_directory, filename)
                    with open(file_path, "r", encoding="utf-8") as f:
                        briefing = json.load(f)
                        self.briefing_list.append(briefing)
                        if briefing["generated_time"].startswith(datetime.today().strftime("%Y-%m-%d")):
                            self.today_briefing = briefing
            
    def generate_briefing(self, link_content_list, mode = "临时模式"):
        # 生成时间戳
        timestamp = int(time.time())
        formatted_time = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # 提取链接和内容
        prompt = "请根据以下内容生成简报， 注意保留所有的链接：\n\n"
        contents = ""
        links = []
        for item in link_content_list:
            prompt += f"{item['link']}\n{item['content']}\n\n"
            links.append(item['link'])
        
        # 调用LLM生成简报内容
        briefing_content = llm_chat_with_prompt(prompt, contents)
        
        # 生成简报字典
        briefing = {
            "generated_time": formatted_time,
            "content": briefing_content,
            "links": links,
            "file_path": ""
        }

        if mode == "临时模式":
            return briefing
        else:
            # 保存简报到文件
            filename = f"briefing_{timestamp}.json"
            file_path = os.path.join(self.save_directory, filename)
            briefing["file_path"] = file_path
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(briefing, f, ensure_ascii=False, indent=4)
            self.today_briefing = briefing
            print("简报内容如下:")
            print(briefing)
            return self.today_briefing

def main():
    save_directory = "briefings"
    briefing_manager = BriefingAgent(save_directory)
    link_content_list = [
        {"link": "https://example.com/article1", "content": "This is the content of article 1."},
        {"link": "https://example.com/article2", "content": "This is the content of article 2."},
        {"link": "https://example.com/article3", "content": "This is the content of article 3."}
    ]
    briefing = briefing_manager.generate_briefing(link_content_list, mode = "简报模式")
    print(briefing)

if __name__ == "__main__":
    main() 