# -*- coding: utf-8 -*-
import random
from urllib.request import urlopen

from bs4 import BeautifulSoup


def get_issue():
    ran_list = []
    ran_num = random.randint(1, 20)

    for i in range(5):
        while ran_num in ran_list:
            ran_num = random.randint(1, 20)
        ran_list.append(ran_num)

    ran_list.sort()

    url = urlopen('https://www.naver.com')
    soup = BeautifulSoup(url, 'html.parser')
    result = soup.select(".PM_CL_realtimeKeyword_rolling span[class*=ah_k]")

    issue_data = list()
    issue_data.extend([u"오늘의 이슈로는"])

    for idx, title in enumerate(result, 1):
        if idx in ran_list:
            issue_data.extend([title.text, ''])

    issue_data.extend(["등 이있습니다."])
    text = ' , '.join(issue_data)
    return text
