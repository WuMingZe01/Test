# 情绪和增长率的关系

#加载情感分析模块
from snownlp import SnowNLP
#from snownlp import sentiment
import pandas as pd
import matplotlib.pyplot as plt
#导入样例数据

# #读取文本数据
# df=pd.read_csv('110022_guba_snownlp_nb.csv')
# #提取所有数据
# df1=df.iloc[:,1]
# print('将提取的数据打印出来：\n',df1)
# #遍历每条评论进行预测
# values=[SnowNLP(i).sentiments for i in df1]
# #输出积极的概率，大于0.5积极的，小于0.5消极的
# #myval保存预测值
# myval=[]
# good=0
# bad=0
# for i in values:
#    if (i>=0.5):
#        myval.append("正面")
#        good=good+1
#    else:
#        myval.append("负面")
#        bad=bad+1
# df['预测值']=values
# df['评价类别']=myval
# #将结果输出到Excel
# df.to_csv('1111.csv')
# rate=good/(good+bad)
# print('好评率','%.f%%' % (rate * 100)) #格式化为百分比


# #作图
# df=pd.read_csv('1111.csv')
# values=df.iloc[:,4]
# y=values
# plt.rc('font', family='SimHei', size=10)
# plt.plot(y, marker='o', mec='r', mfc='w',label=u'评价分值')
# plt.xlabel('用户')
# plt.ylabel('评价分值')
# # 让图例生效
# plt.legend()
# #添加标题
# plt.title('京东评论情感分析',family='SimHei',size=14,color='blue')
# plt.show()

import plotly.graph_objs as go
import streamlit as st

df=pd.read_csv('fund_1.csv')

def get():
    trace1 = go.Scatter(
        x=df['净值日期'],
        y=df['日增长率'],
        name='日增长率',
    )
    trace2 = go.Scatter(
        x=df['净值日期'],
        y=df['预测值'],
        name='情感指数',
        xaxis='x',
        yaxis='y2', # 标明设置一个不同于trace1的一个坐标轴
    )

    data = [trace1, trace2]
    layout = go.Layout(
        yaxis2=dict(anchor='x', overlaying='y', side='right')  # 设置坐标轴的格式，一般次坐标轴在右侧
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_xaxes(
        rangeslider_visible=True,  # 开始显示
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),  # 往前推一个月
                dict(count=6, label="6m", step="month", stepmode="backward"),  # 往前推6个月
                dict(count=1, label="YTD", step="year", stepmode="todate"),  # 只显示今年数据
                dict(count=1, label="1y", step="year", stepmode="backward"),  # 显示过去一年的数据
                dict(step="all")  # 显示全部数据
            ])
        )
    )
    st.plotly_chart(fig)


a = get()
