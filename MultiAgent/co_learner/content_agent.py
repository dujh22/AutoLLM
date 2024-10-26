import json
from typing import List, Dict, Optional
from llm_chat import llm_chat_with_prompt
from datetime import datetime # 时间戳

class ContentAgent:
    def __init__(self):
        """
        初始化内容管理器
        """
        pass    

    def summarize_text(self, text: str, prompt: Optional[str] = None) -> str:
        """
        调用大模型生成摘要
        :param text: 需要摘要的文本
        :param prompt: 用户偏好的提示词
        :return: 摘要结果
        """
        if prompt == None:
            prompt = "你的任务是从给定的文本中生成一个内容的简短摘要。"
        return llm_chat_with_prompt(prompt, text)

    def text_standard(self, link: Optional[str] = None, title: Optional[str] = None, summary: Optional[str] = None, content: Optional[str] = None, user_prompt: Optional[str] = None, keywords: Optional[List[str]] = None) -> Dict:
        """
        将输入内容格式化为标准格式
        :param link: 要解析的网页链接
        :param title: 网页标题
        :param summary: 网页摘要
        :param keywords: 网页关键词
        :return: 内容构成的字典
        """
        timestamp = datetime.now().isoformat()  # 获取当前时间并格式化为ISO格式
        if link == None:
            link = "未知来源"
        if title == None:
            title = self.summarize_text(content, prompt="你的任务是从给定的文本中生成一个内容的标题。")
        if summary == None:
            summary = self.summarize_text(content, prompt="你的任务是从给定的文本中生成一个内容的简短摘要。")
        if keywords == None:
            keywords = self.summarize_text(summary, prompt="你的任务是从给定的文本中生成一个内容的简短关键词列表。并且将关键词列表以空格分隔。").split(" ")
        user_summary = self.summarize_text(content, prompt=user_prompt) if user_prompt else summary
        return {
            "link": link,
            "title": title,
            "summary": summary,
            "user_summary": user_summary,
            "keywords": keywords,
            "timestamp": timestamp,
        }

    def save_content_as_json(self, content: Dict, file_path: str) -> None:
        """
        将内容保存为JSON文件
        :param content: 内容字典
        :param file_path: 保存文件的路径
        """
        with open(file_path, "w", encoding="utf-8") as json_file:
            json.dump(content, json_file, ensure_ascii=False, indent=4)
        print(f"Content saved to {file_path}")

def main():
    # 初始化ContentManager实例
    content_manager = ContentAgent()
    
    # 示例输入
    link = "https://example.com/sample-article"
    title = "Sample Article Title"
    content = "这是一个示例文章的内容。本文介绍了如何使用Python编写一个内容管理器，并生成摘要、标题和关键词。"
    user_prompt = "请根据给定内容提取技术要点。"
    
    # 调用text_standard方法格式化内容
    formatted_content = content_manager.text_standard(
        link=link,
        title=title,
        content=content,
        user_prompt=user_prompt
    )
    
    # 打印生成的内容
    print("Formatted Content:")
    print(json.dumps(formatted_content, ensure_ascii=False, indent=4))
    
    # 将内容保存为JSON文件
    content_manager.save_content_as_json(formatted_content, "sample_content.json")

if __name__ == "__main__":
    main()