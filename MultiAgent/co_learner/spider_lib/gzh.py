# -*- coding: UTF-8 -*-
import requests
import time
import pandas as pd
import math
import random

# List of user agents to randomize requests
user_agent_list = [
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/45.0.2454.85 Safari/537.36 115Browser/6.0.3',
    'Mozilla/5.0 (Macintosh; U; Intel Mac OS X 10_6_8; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-us) AppleWebKit/534.50 (KHTML, like Gecko) Version/5.1 Safari/534.50',
    'Mozilla/5.0 (Windows NT 6.1; rv:2.0.1) Gecko/20100101 Firefox/4.0.1',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_7_0) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11',
    'Mozilla/5.0 (compatible; MSIE 9.0; Windows NT 6.1; Trident/5.0)',
    'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Mobile Safari/537.36',
]

# Target URL and user-specific data
url = "https://mp.weixin.qq.com/cgi-bin/appmsg"
cookie = "ua_id=6fQ22Ya4jEJabkXnAAAAAOJrs1dsShP-vUqte2kmCRE=; wxuin=20840780394661; mm_lang=zh_CN; pgv_pvid=1721521690181522; _qimei_uuid42=187150b291c100715525f319ff3317dfaba3ce5e3d; pac_uid=0_6KAsj1JXFWkFM; _qimei_h38=3d35a5ee5525f319ff3317df03000006c18715; _qimei_q36=; suid=user_0_6KAsj1JXFWkFM; _qimei_fingerprint=c9e4abe78f4ab85dbd304c8b937706b6; _clck=ytda5u|1|fqc|0; uuid=75415a8d3586eb02dfcd848b6af40fa3; rand_info=CAESIAvnXEqNMnh54EyyLAKsR8Dq3m+cDGKrQ6Z9zuDIqSS3; slave_bizuin=3881264596; data_bizuin=3881264596; bizuin=3881264596; data_ticket=99Bx7RXH+KjJrxuG8WCqf6KixwON6yLi8M7Gqt3MA2XfTUEDHqpE/meZ8jCmbal2; slave_sid=VW9MUFBmQmtXaEkwdHkyVG5ZeXNCZEVBaHYzV1JkOE1mQ3Vtdm9EZHc3S2VGOUhBZ0NpVUJIRXJ4TVlaa2RNUWo2ZEM4QU5TQmRfQ3JGd3JjR0tDSGpfTmNyS0pSeDNyd2dUSTkwaHZSWmM4Nmt6UDJXWFJRRlpOcHNXWVd4bEdxYnZ3TnZPZzVTUkROMVZr; slave_user=gh_d02d135b51f8; xid=59370f01f8c1edb738334d19029f81e2; _clsk=kcwth6|1729965230304|2|1|mp.weixin.qq.com/weheat-agent/payload/record"  # Replace with your copied cookie value
fakeid = "MzA3MzI4MjgzMw=="  # Replace with the required fakeid

# Request payload and headers
data = {
    "token": "584931812",
    "lang": "zh_CN",
    "f": "json",
    "ajax": "1",
    "action": "list_ex",
    "begin": "0",
    "count": "5",
    "query": "",
    "fakeid": fakeid,
    "type": "9",
}
headers = {
    "Cookie": cookie,
    "User-Agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/77.0.3865.75 Mobile Safari/537.36",
}

# Initial request to get the total count and calculate the number of pages
try:
    content_json = requests.get(url, headers=headers, params=data).json()
    count = int(content_json["app_msg_cnt"])
    page = int(math.ceil(count / 5))
    print(f"Total articles: {count}, Total pages: {page}")
except Exception as e:
    print(f"Error during initial request: {e}")
    exit()

# Scraping loop
content_list = []
for i in range(page):
    data["begin"] = i * 5
    user_agent = random.choice(user_agent_list)
    headers["User-Agent"] = user_agent

    try:
        # Request page content
        content_json = requests.get(url, headers=headers, params=data).json()
        
        # Parse and save each item
        for item in content_json["app_msg_list"]:
            items = [
                item["title"],
                item["link"],
                time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item["create_time"]))
            ]
            content_list.append(items)

        # Periodic save
        if (i > 0) and (i % 10 == 0):
            test = pd.DataFrame(content_list, columns=['title', 'link', 'create_time'])
            test.to_csv("url.csv", mode='a', encoding='utf-8', index=False, header=not bool(i))  # Only add header once
            print(f"Saved after page {i}")
            content_list = []
            time.sleep(random.randint(60, 90))
        else:
            time.sleep(random.randint(15, 25))

    except Exception as e:
        print(f"Error on page {i}: {e}")
        time.sleep(5)  # Shorter delay on error and continue

# Final save if there are remaining items
if content_list:
    test = pd.DataFrame(content_list, columns=['title', 'link', 'create_time'])
    test.to_csv("url.csv", mode='a', encoding='utf-8', index=False, header=False)
    print("Final save completed")