# 爬取110022易方达消费股吧评论

import requests ##获取网页
from lxml import etree  ##解析文档
import pandas as pd     ##保存文件
import numpy as np



def main(page):
    all_title = []   #爬取的标题存储列表
    all_time  = []   #爬取的发表时间储存列表

    fundcode = 110022    #可替换任意基金代码
#     sleep(random.uniform(1, 2))  #随机出现1-2之间的数，包含小数
    headers = {"User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/93.0.4577.63 Safari/537.36"}
    url = f'http://guba.eastmoney.com/list,of{fundcode}_{page}.html'
    response = requests.get(url, headers=headers, timeout=10)
    #解析网页源代码
    root = etree.HTML(response.text)
    title = root.xpath("//div[@id='articlelistnew']/div/span[@class='l3']/a/text()")
    time = root.xpath("//div[@id='articlelistnew']/div/span[@class='l5']/text()")

    all_title += title  #保存到总数组上
    all_time  += time

    data_title = pd.DataFrame()
    data_time = pd.DataFrame()
    data_title['title'] = all_title
    all_time.remove('最后更新')
    data_time['time'] = all_time
    data_raw =pd.concat([data_title, data_time], axis=1, join='inner')
    print(data_raw)
    data_raw.to_csv('110022_guba.csv', index=False,encoding='utf_8_sig',mode='a')

if __name__ == '__main__':
    for page in range(1,1830): #  最大页数加一
        main(page)
#         time.sleep(random.uniform(1, 2))
        print(f"第{page}页提取完成")