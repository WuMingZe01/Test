# 数据处理
# 基于akshare获取的基金历史信息,保存为fund_Price.csv
# 前期制作好的110022_guba_snownlp_nb.csv
# 将历史信息，周末日期填充，并将空白信息填充为前一日价格，日涨幅为0(辅助工具excel)
# 将填充完的数据和用户情绪打分合并生成最终的数据样本fund_Price_fill.csv
import pandas as pd
import numpy as np
import akshare as ak
# id = 110022
# fund_em_info_df = ak.fund_em_open_fund_info(fund=id, indicator="单位净值走势")
# fund_em_info_df_1 = ak.fund_em_open_fund_info(fund=id, indicator="累计净值走势")
# fund_em_info_df["累计净值"] = fund_em_info_df_1.loc[:, '累计净值']
# print(fund_em_info_df)
# fund = fund_em_info_df.to_csv('fund_Price.csv',encoding='utf_8_sig',index=False)

# df = pd.read_csv('fund_Price.csv',index_col='净值日期')
# df = pd.DataFrame(df)
# df2 = pd.read_csv('fund_Price_fill.csv',index_col='净值日期')
# df2 = pd.DataFrame(df2)
# df1=df[['单位净值','累计净值']]
# #abs缺少数据处理，周末没有交易
# df1 = df1.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
# day = df1.resample('D').asfreq()
# df1 = df1.resample('D').ffill()
# print(df1)
#

# df1=df[['日增长率']]
# #abs缺少数据处理，周末没有交易
# df1 = df1.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
# day = df1.resample('D').asfreq()
# # df1 = df1.resample('D').finall()
# # print(day)
# day = day.to_csv('day.csv', encoding='utf_8_sig')

# df1 = df1.to_csv('fund_Price_fill.csv',encoding='utf_8_sig')

df = pd.read_csv('110022_guba_snownlp_nb.csv',index_col='time')
df = pd.DataFrame(df)
df1=df[['nb_result']]
#abs缺少数据处理，周末没有交易
df1 = df1.set_index(pd.DatetimeIndex(pd.to_datetime(df.index)))
day = df1.resample('D').asfreq()
# df1 = df1.resample('D').finall()
print(day)
day = day.to_csv('day.csv', encoding='utf_8_sig')