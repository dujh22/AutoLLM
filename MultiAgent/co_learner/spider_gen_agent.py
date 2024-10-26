import os
import time
from llm_chat import llm_chat_with_prompt
import subprocess

def run_python(code, timeout=None):
    result = subprocess.run(['python', '-c', code], capture_output=True, text=True, timeout=timeout)
    stdout = result.stdout
    stderr = result.stderr
    return stdout, stderr

class SpiderGenAgent:
    def __init__(self):
        print("3.1 初始化爬虫生成器,校验是否从存在爬虫Agent")
        self.folder = "spider_lib"
        os.makedirs(self.folder, exist_ok=True)
    
    def get_parser_paths(self, link: str):
        if link.startswith("http://mp.weixin.qq.com/"):
            return [os.path.join(self.folder, "gzh.py"), os.path.join(self.folder, "url_to_content_to_abstrut.py")]
        elif link.startswith("http://arxiv.org/"):
            return [os.path.join(self.folder, "arx.py"), os.path.join(self.folder, "arx_to_ch.py")]
        else:
            return self.generate_parser(link)
    
    def generate_parser(self, link: str, max_attempts: int = 256):
        for attempt in range(max_attempts):
            prompt = "你是一个爬虫专家，你擅长从网页中提取信息。现在开始，请生成一个爬虫来提取这个网页的内容。"
            code = llm_chat_with_prompt(prompt, link)  # Generate code using LLM
            script_name = f"scraper_{int(time.time())}.py"
            script_path = os.path.join(self.folder, script_name)
            
            with open(script_path, 'w') as file:
                file.write(code)
                
            if self.validate_script(script_path, link):  # Custom method to validate generated script
                return [script_path]
            os.remove(script_path)
        
        raise ValueError("Failed to generate a functional scraper after max attempts.")
    
    def validate_script(self, script_path, link):
        # 定义一个简单的验证逻辑，比如运行查看是否有输出
        # 运行python 程序
        code = ""
        with open(script_path, 'r') as file:
            code = file.read()
        stdout, stderr = run_python(code)
        return stdout, stderr

def main():
    # Usage example
    agent = SpiderGenAgent()
    print(agent.get_parser_paths("http://mp.weixin.qq.com/some-article"))

if __name__ == "__main__":
    main()