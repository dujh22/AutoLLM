from datetime import datetime
import json
from spider_gen_agent import SpiderGenAgent
from info_get_agent import InfoGetAgent
from info_filter_agent import InfoFilterAgent
from briefing_agent import BriefingAgent
import os
from llm_chat import llm_chat_with_prompt
import time

class Agent3_InfoFilterAgent:
    def filter_links(self, link_list, history_briefing):
        # 筛选出符合条件的link（简化逻辑）
        filtered_links = [link for link in link_list if "relevant" in link]
        return filtered_links

class AutoAgent:
    def __init__(self, userid, user_preference):
        '''
        user_id: 用户id
        '''
        self.user_preference = user_preference
        self.agent1 = SpiderGenAgent()
        self.agent2 = InfoGetAgent(userid)
        self.userpath = self.agent2.user_folder
        self.briefagent = BriefingAgent(userid)

    def execute(self, link):
        if self.briefagent.today_briefing is None:
            print("今日简报不存在，生成新的简报")
            link_list = []
            link_abs_list = []
            link_title_list = []
            print("3.3 从用户知识库中获取所有相关链接")
            # 读取userpath下每一个json文件，获得对应的link，contents
            for file in os.listdir(self.userpath):
                if file.endswith(".json"):
                    with open(os.path.join(self.userpath, file), "r") as f:
                        data = json.load(f)
                        link_list.append(data["link"])
                        link_abs_list.append(data["summary"])
                        link_title_list.append(data["title"])
            print("3.4 从用户知识库中获取用户偏好")
            agent = InfoFilterAgent(link_list, link_abs_list, self.user_preference)
            top_contents = agent.get_top_n_contents(10)
            print("3.5 生成简报")
            print("3.6 生成可供分享的语音和数字人播报视频")
            return self.briefagent.generate_briefing(top_contents, mode="正式模式")
        else:
            print("今日简报已存在，结合link信息生成新的简报")
            prompt = "请根据以下内容生成简报， 注意保留所有的链接：\n\n"
            today_briefing = self.briefagent.today_briefing
            link_name = self.agent2.get_file_name(link)
            contents = ""
            temp_link = ""
            path = os.path.join(self.userpath, f"{link_name}.json")
            with open(path, "r") as f:
                data = json.load(f)
                contents = data["summary"]
                temp_link = data["link"]
            briefing_content = llm_chat_with_prompt(prompt, today_briefing["content"] + contents)
            new_briefing = {
                "generated_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "content": briefing_content,
                "links": today_briefing["links"] + [temp_link],
                "file_path": ""
            }
            print("3.5 生成临时简报")
            print("简报内容如下:")
            print(new_briefing)
            print("3.6 生成可供分享的语音和数字人播报视频")
            return new_briefing
