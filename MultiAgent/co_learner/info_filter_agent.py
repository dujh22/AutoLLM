import random
from llm_chat import llm_chat_with_prompt
import time

class InfoFilterAgent:
    def __init__(self, link_list, link_contents_list, user_preference):
        '''
        link_list: 链接列表
        link_contents_list: 链接内容列表
        user_preference: 用户偏好
        '''
        print("3.4 语义筛选与智能排序")
        self.link_list = link_list
        self.contents = link_contents_list  # 步骤1：获取内容,格式上应该是字典，key是link，value是content
        self.user_preference = user_preference

    def score_content(self, content):
        # 模拟评分系统，模型打分
        # 重要性打分
        importance_prompt = "请根据内容的重要性打分，范围是0-100，越高表示越重要。只需要返回一个整数。"
        importance_score = llm_chat_with_prompt(importance_prompt, content)
        if not importance_score.isdigit():
            importance_score = 0
        # 用户偏好打分
        preference_prompt = "用户的偏好是：" + self.user_preference + "请根据用户的偏好打分，范围是0-100，越高表示越偏好。只需要返回一个整数。"
        preference_score = llm_chat_with_prompt(preference_prompt, content)
        if not preference_score.isdigit():
            preference_score = 0
        # 总得分
        total_score = int(importance_score) + int(preference_score)
        return total_score

    def rank_contents(self):
        # 步骤2：对每个内容打分并排序
        scored_contents = []
        for link, content in zip(self.link_list, self.contents):
            score = self.score_content(content)
            print(f"Link: {link}, Content: {content}, Score: {score}")
            # 等待1秒钟
            time.sleep(1) # 模拟耗时操作
            scored_contents.append({"link": link, "content": content, "score": score})
            

        # 排序：按得分降序排列
        ranked_contents = sorted(scored_contents, key=lambda x: x["score"], reverse=True)
        return ranked_contents

    def get_top_n_contents(self, n=10):
        # 步骤3：返回前N个内容
        ranked_contents = self.rank_contents()
        top_n_contents = ranked_contents[:n]
        return top_n_contents

def main():
    # 使用示例
    link_list = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8", "link9", "link10", "link11"]
    content_list = ["content1", "content2", "content3", "content4", "content5", "content6", "content7", "content8", "content9", "content10", "content11"]
    user_preference = "user_preference"
    agent = InfoFilterAgent(link_list, content_list, user_preference)
    top_contents = agent.get_top_n_contents(10)

    # 打印结果
    for i, content in enumerate(top_contents, start=1):
        print(f"Rank {i}: Link: {content['link']}, Score: {content['score']}, Content: {content['content']}")

if __name__ == "__main__":
    main()