import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import akshare as ak
import plotly_express as px
from PIL import Image


st.sidebar.subheader('开放式基金相关信息')
image = Image.open('img/img.jpg')
image1 = Image.open('img/img.jpeg')
st.sidebar.image(image, use_column_width=True)
choose = st.sidebar.radio(
"请选择您要查看的内容：",
('基金查询','基金经理', '基金排行'))
if choose == '基金查询':
    st.image(image1, use_column_width=True)
    id = st.text_input('基金ID：')
    if not id:
        st.warning('请输入您要查询的基金ID.')
        st.stop()
    st.success(f'欢迎查看基金:{id}')

    st.subheader('1.读入数据')

    fund_em_info_df = ak.fund_em_open_fund_info(fund=id, indicator="单位净值走势")
    fund_em_info_df_1 = ak.fund_em_open_fund_info(fund=id, indicator="累计净值走势")
    fund_em_info_df["累计净值"] = fund_em_info_df_1.loc[:, '累计净值']

    fund_em_info_df_2 = ak.fund_em_open_fund_info(fund=id, indicator="累计收益率走势")
    fund_em_info_df["累计收益率"] = fund_em_info_df_2.loc[:, '累计收益率']

    fund_em_info_df_3 = ak.fund_em_open_fund_info(fund=id, indicator="同类排名走势")
    fund_em_info_df["同类型排名-每日近三月排名"] = fund_em_info_df_3.loc[:, '同类型排名-每日近三月排名']

    fund_em_info_df_4 = ak.fund_em_open_fund_info(fund=id, indicator="同类排名走势")
    fund_em_info_df["总排名-每日近三月排名"] = fund_em_info_df_4.loc[:, '总排名-每日近三月排名']

    fund_em_info_df_5 = ak.fund_em_open_fund_info(fund=id, indicator="同类排名百分比")
    fund_em_info_df["同类型排名-每日近3月收益排名百分比"] = fund_em_info_df_5.loc[:, '同类型排名-每日近3月收益排名百分比']

    fund = fund_em_info_df

    fund = fund.to_csv('fund.csv', index=None)
    df = pd.read_csv('fund.csv')

    st.dataframe(df, width=930, height=330)

    st.subheader('2.基金数据可视化')


    def get_jingzhi():
        fig = px.area(
            df,
            x='净值日期',
            y=['单位净值', '累计净值'],
            title='基金净值走势图',
            color_discrete_map={'累计净值': '#b7e981', '单位净值': '#636ffa'},
            width=930,
            height=630)

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


    def get_rizengzhang():
        fig = px.scatter(
            df,
            x='净值日期',
            y=['日增长率'],
            title='日增长率',
            color_discrete_map={'日增长率': '#66c6cd'},
            width=930,
            height=630
        )

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


    def get_shouyi():
        df1 = ak.fund_em_open_fund_info(fund=id, indicator="累计收益率走势")
        fig = px.line(
            df1,
            x='净值日期',
            y=['累计收益率'],
            title='近三月累计收益率',
            color_discrete_map={'累计收益率': 'red'},
            width=930,
            height=630
        )

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


    def get_paiming():
        fig = px.line(
            df,
            x='净值日期',
            y=['同类型排名-每日近三月排名', '总排名-每日近三月排名', '同类型排名-每日近3月收益排名百分比'],
            title='基金净值走势图',
            width=930,
            height=630)

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


    def get_chicang():
        fund_em_portfolio_hold_df = ak.fund_em_portfolio_hold(code=id, year="2021")
        fund_stock = fund_em_portfolio_hold_df.head(8)

        fig = px.sunburst(fund_stock,
                          path=['股票名称', '占净值比例', '持股数'],
                          values='占净值比例',
                          title='基金股票持仓比例图',
                          width=730,
                          height=730
                          )

        st.plotly_chart(fig)


    data_base = get_jingzhi()
    data_base1 = get_rizengzhang()
    data_base2 = get_shouyi()
    data_base3 = get_paiming()
    data_base4 = get_chicang()


elif choose == '基金排行':
    st.subheader('查看基金排行：')
    image2 = Image.open('img/img3.jpg')
    st.image(image2, use_column_width=True)
    fund_em_open_fund_rank_df = ak.fund_em_open_fund_rank(symbol="全部")
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.drop(['序号'],axis=1)
    st.write(fund_em_open_fund_rank_df)

elif choose == '基金经理':
    st.subheader('基金经理信息：')
    manager = pd.read_csv('fund_manager.csv')
    manager.sort_values("现任基金资产总规模", inplace=True, ascending=False)
    manager_guimo = manager.head(10)

    def get_guimo():
        fig = px.funnel(
            manager_guimo,
            x='现任基金资产总规模',
            y='姓名',
            title='基金经理规模排行（单位：亿）',
            color='现任基金资产总规模'
            )

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





    manager.sort_values("现任基金最佳回报", inplace=True, ascending=False)
    manager_huibao = manager.head(20)

    def get_huibao():
        fig = px.bar(
            manager_huibao,
            x='姓名',
            y='现任基金最佳回报',
            title='现任基金最佳回报_前二十（单位：%）',
            color='现任基金最佳回报',

        )

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



    manager.sort_values("累计从业时间", inplace=True, ascending=False)
    manager_time = manager

    def get_time():
        fig = px.histogram(
            manager_time,
            x='累计从业时间',
            # y=['10','20','30'],
            title='累计从业时间比例（单位：年）',
            color='累计从业时间',

        )

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

    manager1 = get_guimo()
    manager2 = get_huibao()
    manager3 = get_time()




