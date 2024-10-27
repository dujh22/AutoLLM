import os
import json
from datetime import datetime
from auto_agent import AutoAgent
from llm_chat import llm_chat_with_his

class UserManager:
    def __init__(self):
        self.project_dir = os.path.join(os.getcwd(), "user_data")
        self.user_id_list = []
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        else:
            # 获得用户id，即project_dir文件夹下存在的文件夹名
            self.user_id_list = os.listdir(self.project_dir)
            print(f"Existing users: {self.user_id_list}")
        
    # 判断用户是否存在
    def user_exists(self, user_id):
        if user_id in self.user_id_list:
            return True
        else:
            return False

    # 创建用户
    def create_user(self, user_id):
        user_folder = os.path.join(self.project_dir, user_id)
        os.makedirs(user_folder, exist_ok=True)
        # 创建简报文件夹
        os.makedirs(os.path.join(user_folder, "briefings"), exist_ok=True)
        # 创建源文件文件夹
        os.makedirs(os.path.join(user_folder, "source_files"), exist_ok=True)

        # 创建JSONL文件
        # links.jsonl
        with open(os.path.join(user_folder, "news_links.jsonl"), "w") as file:
            file.write("")
        with open(os.path.join(user_folder, "gzh_links.jsonl"), "w") as file:
            file.write("")
        with open(os.path.join(user_folder, "paper_links.jsonl"), "w") as file:
            file.write("")
        # preferences.json
        with open(os.path.join(user_folder, "preferences.json"), "w") as file:
            json.dump([], file)
        # qa_interactions.json
        with open(os.path.join(user_folder, "qa_interactions.json"), "w") as file:
            json.dump({"messages": []}, file)
        
        print(f"User {user_id} created with folders and JSON files.")
    
    # 重置链接
    def reset_link(self, user_id, links, mode):
        time_stamp = datetime.now().isoformat()
        file_path = ""
        if mode == "news":
            file_path = os.path.join(self.project_dir, user_id, "news_links.jsonl")
        elif mode == "gzh":
            file_path = os.path.join(self.project_dir, user_id, "gzh_links.jsonl")
        elif mode == "paper":
            file_path = os.path.join(self.project_dir, user_id, "paper_links.jsonl")
        # 将链接写入jsonl文件中，逐行是一个json
        with open(file_path, "w") as file:
            for link in links:
                file.write(json.dumps({"link": link, "time": time_stamp}) + "\n")
        
    # 获得链接
    def get_links(self, user_id, mode):
        if mode == "news":
            file_path = os.path.join(self.project_dir, user_id, "news_links.jsonl")
        elif mode == "gzh":
            file_path = os.path.join(self.project_dir, user_id, "gzh_links.jsonl")
        elif mode == "paper":
            file_path = os.path.join(self.project_dir, user_id, "paper_links.jsonl")
        with open(file_path, "r") as file:
            # file每一行是{'link': 'http://example.com/news', 'time': '2024-10-26T21:24:59.623529'}
            lines = file.readlines()
            links = []
            for line in lines:
                line_json = json.loads(line)
                links.append(line_json["link"])
            return links

    # 重制用户偏好为新传入的prompt
    def reset_preference(self, user_id, prompt):
        file_path = os.path.join(self.project_dir, user_id, "preferences.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump([prompt], file)
    
    # 获得用户偏好
    def get_preference(self, user_id):
        file_path = os.path.join(self.project_dir, user_id, "preferences.json")
        with open(file_path, "r", encoding="utf-8") as file:
            files = json.load(file)
            return files[0]

    # 添加用户交互
    def add_interaction(self, user_id, user_message, assistant_message):
        interaction_entry = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message}
        ]
        file_path = os.path.join(self.project_dir, user_id, "qa_interactions.json")
        with open(file_path, "r+") as file:
            data = json.load(file)
            data["messages"].extend(interaction_entry)
            file.seek(0)
            json.dump(data, file)
    
    # 和用户交互
    def interact(self, user_id, user_message):
        file_path = os.path.join(self.project_dir, user_id, "qa_interactions.json")
        return_message_with_link = ""
        return_message_without_link = ""
        with open(file_path, "r") as file:
            data = json.load(file)
            data["messages"].append({"role": "user", "content": user_message})
            print("1. 识别用户输入为link还是prompt。。。")
            if "http" in user_message:
                print("2. 识别到用户输入为link，开始执行简报模式")
                jianbao = self.create_briefing(user_id, user_message)
                return_message_with_link = jianbao
                return_message_without_link = jianbao["content"]
                data["messages"].append({"role": "assistant", "content": return_message_with_link})
            else:
                print("2. 识别到用户输入为prompt，开始执行对话模式")
                llm_response = llm_chat_with_his(data["messages"])
                data["messages"].append({"role": "assistant", "content": llm_response})
                return_message_without_link = llm_response
                return_message_with_link = llm_response
        with open(file_path, "w") as file:
            json.dump(data, file)
        return return_message_with_link, return_message_without_link

    # 周期性创建简报 &  临时性创建简报
    def create_briefing(self, user_id, link):
        user_preference = self.get_preference(user_id)
        auto_agent = AutoAgent(user_id, user_preference)
        briefing = auto_agent.execute(link)
        return briefing
    
    # 获得简报
    def get_briefing(self, user_id):
        directory_path = os.path.join(self.project_dir, user_id, "briefings")
        briefings = ""
        for file_name in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file_name)
            if os.path.isfile(file_path):
                with open(file_path, 'r') as file:
                    briefing_content = json.load(file)
                    briefings = briefing_content
                    break
        return briefings

def main():
    # 示例使用
    user_manager = UserManager()

    # user_manager.interact("134", "http://arxiv.org/abs/2410.18923v1")
    user_manager.user_exists("134")

    # # 创建用户
    # user_id = "2"
    # user_manager.create_user(user_id)

    # # 添加 link 和偏好
    # user_manager.reset_link(user_id, ["https://example.com"], "news")
    # user_manager.reset_preference(user_id, "This is a preferred prompt")

    # # 添加用户交互记录
    # user_manager.add_interaction(user_id, "9.11 和 9.8 谁大？", "9.11 大")

    # # # 获取用户数据
    # print(user_manager.get_links("2", "news"))
    # print(user_manager.get_links("2", "gzh"))
    # print(user_manager.get_links("2", "paper"))
    # print(user_manager.get_preference("2"))

    # print(user_manager.user_id)

    # # 保存源数据
    # source_data = {"data": "source data example"}
    # user_manager.save_source_data(user_id, source_data, "source_name_example")

    # # 使用 AutoAgent 生成定期简报
    # auto_agent = AutoAgent(user_manager, user_id)
    # briefing = auto_agent.generate_periodic_briefing()
    # print("Generated Briefing:", briefing)

if __name__ == "__main__":
    main()