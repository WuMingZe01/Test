#基金评论文本情感分析
# 先使用snownlp库，可直接进行文本情感分析
import pandas as pd
import numpy as np
from snownlp import SnowNLP

df = pd.read_csv("110022_merge.csv")
# df['time'] = pd.to_datetime(df['time']) #将数据类型转换为日期类型
df = pd.DataFrame(df)


def snow_result(comemnt):
    s = SnowNLP(comemnt)
    a = round(s.sentiments,1)
    return a
    # else:
    #     return 0


df['snlp_result'] = df.title.apply(snow_result)
df = df.to_csv('110022_guba_snownlp_fl.csv',encoding='utf_8_sig',index=False)
print(df)
