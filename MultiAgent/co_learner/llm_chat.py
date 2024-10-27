from openai import OpenAI
from dotenv import load_dotenv
import os
import config

# 加载 .env 文件
load_dotenv()

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

# 非流式调用
def llm_chat(prompt, model = config.model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

# 非流式调用
def llm_chat_with_his(messages, model = config.model):
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
    )
    return chat_completion.choices[0].message.content

# 带提示的对话,非流式调用
def llm_chat_with_prompt(prompt, user_str, model = config.model):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": user_str,
            }
        ],
        model=model,
    )
    return chat_completion.choices[0].message.content

# 流式调用
def llm_chat_stream(prompt, model = config.model):
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user", 
                "content": prompt
            }
        ],
        stream=True,
    )
    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")

# 将链接嵌入到文本中，形成html格式数据
def strAndLink(briefing):
    promt = "请重新规整中文简报格式，格式按照html的形式存在。具体的链接应该以<a>标签的形式存在，并且链接的文本应该和链接的地址一致。"
    link_str = ""
    for link in briefing["links"]:
        link_str += f"<a href='{link}'>{link}</a>\n"
    return llm_chat_with_prompt(promt, briefing["content"] + link_str)

# 主要测试函数
def main():
    # print(llm_chat("你好，请问你是谁？"))
    # llm_chat_stream("你好，请问你是谁？")
    # print(llm_chat_with_prompt("你现在是一个医生健康助手", "请问你可以做什么"))

    import json
    with open("/root/dujinhua/co_learner/user_data/134/briefings/briefing_1729987853.json", "r") as f:
        data = json.load(f)
        print(strAndLink(data))

if __name__ == "__main__":
    main()
    