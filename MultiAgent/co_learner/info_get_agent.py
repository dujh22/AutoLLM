import os
import csv
import json
import subprocess
from datetime import datetime
from content_agent import ContentAgent
from tqdm import tqdm

class InfoGetAgent:
    def __init__(self, phone_number):
        print("3.2 爬虫Agent获取最新数据和关联数据")
        self.phone_number = phone_number
        # 获取根目录
        self.user_folder = f"./user_data/{phone_number}/source_files"
        os.makedirs(self.user_folder, exist_ok=True)
        self.content_agent = ContentAgent()

    def check_and_run_crawlers(self, link_crawler_paths):
        for link, crawler_path in link_crawler_paths:
            csv_path = os.path.join(self.user_folder, f"{self.get_file_name(link)}.csv")

            # 检查 CSV 文件是否已经存在
            if not os.path.exists(csv_path):
                print(f"运行爬虫以获取 {link} 的数据...")
                # 运行爬虫脚本
                self.run_crawler(crawler_path, csv_path)
            
            # 将 CSV 文件转换为 JSON 格式
            self.convert_csv_to_json(csv_path, link)

    def run_crawler(self, crawler_path, csv_path):
        try:
            # 调用指定的爬虫脚本，并传递输出路径参数
            subprocess.run(["python", crawler_path, csv_path], check=True)
        except subprocess.CalledProcessError as e:
            print(f"运行爬虫失败: {e}")

    def convert_csv_to_json(self, csv_path, link):
        # 判断link是否包含其中
        flag = False
        link_path = os.path.join(self.user_folder, f"{self.get_file_name(link)}.json")

        with open(csv_path, "r", encoding="utf-8") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            # 将标题改为Title
            if "标题" in csv_reader.fieldnames:
                csv_reader.fieldnames[csv_reader.fieldnames.index("标题")] = "Title"
            # 将摘要改为Summary
            if "摘要" in csv_reader.fieldnames:
                csv_reader.fieldnames[csv_reader.fieldnames.index("摘要")] = "Summary"
            # 逐行处理 CSV，并转换为 JSON
            for row in tqdm(csv_reader, desc="Converting to JSON"):
                temp_json = row
                temp_dict = self.content_agent.text_standard(
                    link = temp_json["URL"],
                    title = temp_json["Title"],
                    summary = temp_json["Summary"],
                )
                temp_path = os.path.join(self.user_folder, f"{self.get_file_name(temp_json['URL'])}.json")
                self.content_agent.save_content_as_json(temp_dict, temp_path)
    
        print(f"已将 {csv_path} 转换为 JSON 文件")

    def get_file_name(self, link):
        """生成基于链接的唯一文件名"""
        return link.replace("http://", "").replace("/", "_").replace(":", "_")

# 使用示例
if __name__ == "__main__":
    link_crawler_paths = [
        ("http://example.com/page1", "spider_lab/spider1.py"),
        ("http://example.com/page2", "spider_lab/spider2.py"),
    ]
    phone_number = "134"
    agent = InfoGetAgent(phone_number)
    # agent.check_and_run_crawlers(link_crawler_paths)
    # agent.convert_csv_to_json("./spider_lib/arxiv_papers_ch.csv", "http://arxiv.org/abs/2410.18923v1")
    agent.convert_csv_to_json("./spider_lib/url_with_summaries.csv", "http://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650940508&idx=1&sn=1483718d500bc76557d5397827638c7f&chksm=84e7e022b3906934994976f420e96f5aaf6a7d87bd799d96b354e167253833e2f445186d5cad#rd")