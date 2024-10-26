import csv
from tqdm import tqdm
import sys
import os

# 将上一级目录添加到系统路径
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from llm_chat import llm_chat_with_prompt


prompt = "你是一个翻译家，你擅长把英文翻译成中文。现在开始， 每次你收到一个英文句子或者段落，都把它翻译成地道的中文。"

# 读取原始CSV文件，并将每一行的需要翻译的字段进行中文翻译
def translate_csv(input_file, output_file):
    # 打开原始文件
    with open(input_file, mode='r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        # 获取CSV的字段名称
        fieldnames = reader.fieldnames

        # 打开用于保存翻译结果的新CSV文件
        with open(output_file, mode='w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            # 遍历每一行，翻译特定字段
            for row in tqdm(reader, desc="Translating"):
                # 翻译Title, Summary, First Author等需要翻译的字段
                row['Title'] = llm_chat_with_prompt(prompt, row['Title'])
                row['Summary'] = llm_chat_with_prompt(prompt, row['Summary'])
                row['First Author'] = llm_chat_with_prompt(prompt, row['First Author'])

                # 将翻译结果写入新CSV文件
                writer.writerow(row)

# 使用示例
input_csv = 'arxiv_papers_raw.csv'
output_csv = 'arxiv_papers_ch.csv'
translate_csv(input_csv, output_csv)