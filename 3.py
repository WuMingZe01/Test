import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import akshare as ak
import plotly_express as px
from PIL import Image
import os
import time
import sys
import csv


def compare(code, years, s1, s2):
    """
    code,str,基金代码;
    years,list,年份列表,['yr1','yr2','……'];
    s1,str,靠前的季度, 格式为 'YYYYQ1',例如: '2021Q2';
    s2,str,靠后的季度, 格式为 'YYYYQ1',例如: '2021Q2';
    注意，s1和s2的年份应在 years 里
    """

    s1_share = s1 + '持股数'
    s2_share = s2 + '持股数'
    s1_value = s1 + '持仓市值'
    s2_value = s2 + '持仓市值'
    s1_ratio = s1 + '持仓比例'
    s2_ratio = s2 + '持仓比例'

    data = pd.DataFrame()
    for yr in years:
        df_tmp = ak.fund_em_portfolio_hold(code=code, year=yr)
        data = data.append(df_tmp)

    data['季度'] = data['季度'].apply(lambda x: x[:6])
    data['季度'] = data['季度'].str.replace('年', 'Q')
    data['占净值比例'] = pd.to_numeric(data['占净值比例'])

    df1 = data[data['季度'] == s1]
    df1 = df1[['股票代码', '股票名称', '持股数', '持仓市值', '占净值比例']]
    df1 = df1.rename(columns={'持股数': s1_share, '持仓市值': s1_value, '占净值比例': s1_ratio})
    df2 = data[data['季度'] == s2]
    df2 = df2[['股票代码', '股票名称', '持股数', '持仓市值', '占净值比例']]
    df2 = df2.rename(columns={'持股数': s2_share, '持仓市值': s2_value, '占净值比例': s2_ratio})

    df_merge = pd.merge(df1, df2, on='股票代码', how='outer')

    # Q2 和 Q4，即半年度和年度报告，是需要披露全部持仓的
    # 合并后，在dataframe 中 NaN 的数据应为 0

    if s1.endswith('Q2') or s1.endswith('Q4'):
        df_merge[s1_share] = df_merge[s1_share].fillna(0)
        df_merge[s1_value] = df_merge[s1_value].fillna(0)
        df_merge[s1_ratio] = df_merge[s1_ratio].fillna(0)

    if s2.endswith('Q2') or s2.endswith('Q4'):
        df_merge[s2_share] = df_merge[s2_share].fillna(0)
        df_merge[s2_value] = df_merge[s2_value].fillna(0)
        df_merge[s2_ratio] = df_merge[s2_ratio].fillna(0)

    df_merge['持股数变化'] = df_merge[s2_share] - df_merge[s1_share]
    df_merge = df_merge.sort_values(s2_value, ascending=False)

    df_merge['股票名称'] = df_merge['股票名称_y']
    # df_merge['股票名称'] = df_merge['股票名称'].fillna('0')
    # df_merge.loc[df_merge['股票名称']=='0','股票名称'] = df_merge.loc[df_merge['股票名称']=='0','股票名称_x']
    df_merge.loc[df_merge['股票名称'].isna(), '股票名称'] = df_merge.loc[df_merge['股票名称'].isna(), '股票名称_x']
    df_merge = df_merge[['股票代码', '股票名称', s1_share,
                         s1_value, s1_ratio,
                         s2_share, s2_value,
                         s2_ratio, '持股数变化']]
    df_merge.head(10)
    return df_merge

compare(110022,[2021],'2021Q2','2021Q1')