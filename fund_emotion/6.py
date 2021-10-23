# 从业时间(天)-管理资产规模(亿)-最佳回报(%)对比
# 3D散点图
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly_express as px
import streamlit as st
df = pd.read_csv("manager_1.csv")
df = pd.DataFrame(df)

# def get():
fig = px.scatter_3d(df,
                    x='累计从业时间',
                    y='现任基金资产总规模',
                    z='现任基金最佳回报',
                    color='现任基金最佳回报',
                    # size_max=18,
                    # opacity=0.7,
                    title='从业时间(天)-管理资产规模(亿)-最佳回报(%)对比'
                    )
fig.show()
# a = get()

