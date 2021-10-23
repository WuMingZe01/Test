# 数据清洗，清洗掉广告公告等数据
# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("110022_guba.csv",names=['title', 'time'])
# # df['time'] = pd.to_datetime(df['time']) #将数据类型转换为日期类型
# df = pd.DataFrame(df)
#
# df['time'] = df['time'].map(lambda x:x.split(' ')[0])
# df = df[ ~ df['title'].str.contains('易基消费|易方达消费行业股票型|易方达基金管理有限公司|我在20|【|#|title') ]
# df = df.to_csv('clean.csv',encoding='utf_8_sig',index=False)
# print(df)


# 数据合并，将每日评论合并
import pandas as pd
import numpy as np

df = pd.read_csv("110022_clean.csv")
#定义拼接函数，并对字段进行去重
def concat_func(x):
    return pd.Series({
        'title':','.join(x['title'].unique())
    }
    )
#分组聚合+拼接
result=df.groupby(df['time']).apply(concat_func).reset_index()
result = result.sort_values(by='time',ascending=False)
result = result.reset_index(drop = True)
result = result.to_csv('110022_merge.csv',encoding='utf_8_sig',index=False)
#结果展示
print(result)