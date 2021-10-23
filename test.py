import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import plotly.figure_factory as ff
import akshare as ak
import plotly_express as px
from PIL import Image
import csv


st.sidebar.subheader('开放式基金相关信息')
image = Image.open('img/img.jpg')
image1 = Image.open('img/img.jpeg')
st.sidebar.image(image, use_column_width=True)
choose = st.sidebar.radio(
"请选择您要查看的内容：",
('基金查询','基金经理', '基金排行','市场情绪'))
if choose == '基金查询':
    st.image(image1, use_column_width=True)
    mydict = {}
    with open('fund_name.csv', mode='r', encoding='utf_8') as inp:
        reader = csv.reader(inp)
        dict_from_csv = {rows[0]: rows[1] for rows in reader}
    id = st.text_input('基金ID：')
    name = dict_from_csv.get(id)
    if not id:
        st.warning('请输入您要查询的基金ID.')
        st.stop()
    st.success(f'欢迎查看基金:{name}（{id}）')

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

    fund = fund.to_csv('fund.csv', encoding='utf_8',index=None)
    df = pd.read_csv('fund.csv')

    st.dataframe(df, width=930, height=330)

    st.subheader('2.基金数据可视化')

    # 净值
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

    # 日增长率
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

    #收益率
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

    # 排名情况
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

    #持仓比例
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


    # 数据爬取及预测模型
    st.subheader('3.基金涨幅预测')
    import requests
    import time
    import execjs

    st.markdown(
        """
        <style>
            .stProgress > div > div > div > div {
                background-image: linear-gradient(to right, #99ff99 , #00ccff);
            }
        </style>""",
        unsafe_allow_html=True,
    )
    my_bar = st.progress(10)

    with st.spinner('正在进行预测，请耐心等待...'):
        time.sleep(5)

    # LSTM，多特征值预测基于用户情感倾向

    import requests
    from typing import Tuple
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential
    import matplotlib.pyplot as plt
    from urllib.parse import urlencode
    fig = plt.figure(figsize=(15, 6))

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    my_bar.progress(20)
    def split_sequences(X: np.ndarray, time_steps: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        '''
        生成输入输出序列

        参数
        ---
        X : m 行 n 列 array ，其中最后一列为待输出的输出，前面的均为待输入的数据
        time_steps : 时间步长，即使用前 time_steps 给数据来预测下一个数据

        返回
        ---
        生成的 X 和 y array

        '''
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        length = len(X)
        n_features = X.shape[1] - 1
        _X = []
        _y = []
        for start_idx in range(length):
            end_idx = start_idx + time_steps
            if end_idx >= length:
                break
            _X.append(X[start_idx:end_idx, :-1])
            _y.append(X[end_idx - 1, -1])
        _X = np.array(_X)
        _X = _X.reshape((-1, time_steps, n_features))
        _y = np.array(_y)
        return (_X, _y)


    # 读取股票信息表格
    df = pd.read_csv('./fund_emotion/data_merge_lstm/fund_Price_fill.csv')
    # 删除缺少值
    df = df.dropna()
    my_bar.progress(30)
    # 指定研究的特征（最后一个为待预测的数据!）
    features = ['累计净值', '情绪指数', '日增长率', '单位净值']

    # 创建一个 MinMaxScaler 对象，方便将数据归一化
    # 归一化公式：x =  (x - min)/(max-min)
    sc = MinMaxScaler(feature_range=(0, 1))

    # 将数据转为 numpy 数组
    values = df[features].to_numpy()

    # 归一化 values
    scaled_values = sc.fit_transform(values)

    # 生成序列
    X, y = split_sequences(scaled_values)

    # 训练数据个数
    train_size = int(len(df) * 0.6)

    # 训练集
    X_train = X[:train_size]
    y_train = y[:train_size]

    # 测试集
    X_test = X[train_size:]
    y_test = y[train_size:]

    my_bar.progress(60)

    # 创建模型
    model = Sequential()
    # 添加 含有 50 个单元的 LSTM 网络(第一层)
    model.add(LSTM(50, return_sequences=True))
    # 添加 含有 30 个单元的 LSTM 网络(第二层)
    model.add(LSTM(30, return_sequences=True))
    # 添加 含有 10 个单元的 LSTM 网络(第三层)
    # 注意，最后一层没有 return_sequences = True ！！！
    model.add(LSTM(10))

    # 添加输出层网络以输出预测的股票收盘价格
    model.add(Dense(1))
    # 编译模型
    model.compile(loss='mae', optimizer='adam')
    # 拟合模型
    model.fit(X_train, y_train, epochs=30, validation_split=0.2)

    my_bar.progress(80)
    # 真实收盘价格
    y_real = values[-len(y_test):, -1]
    plt.plot(y_real, label='真实基金价格')

    y_p = model.predict(X_test).reshape(-1, 1)
    # 将全部收盘价归一化
    sc.fit_transform(df['单位净值'].to_numpy().reshape(-1, 1))

    # 归一化逆过程，即将归一化的数据转为真实数据
    # 预测的收盘价格
    y_p = sc.inverse_transform(y_p.reshape(-1, 1))
    # 绘图

    plt.plot(y_p, label='预测基金价格')
    # 绘制图例
    plt.legend()
    # 绘制标题
    plt.title('根据 {} 预测 {} 的基金走势'.format("、".join(features[:-1]), 110022))
    # 保存图片
    plt.savefig(f'110022.jpg')
    # 显示图像
    plt.show()

    my_bar.progress(100)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)
    st.success('Done!')
    st.balloons()
    st.pyplot(fig)


elif choose == '基金排行':
    st.subheader('查看基金排行：')
    image2 = Image.open('img/img3.jpg')
    st.image(image2, use_column_width=True)
    fund_em_open_fund_rank_df = ak.fund_em_open_fund_rank(symbol="全部")
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.drop(['序号'],axis=1)
    st.write(fund_em_open_fund_rank_df)

    fund_em_open_fund_rank_df = pd.DataFrame(fund_em_open_fund_rank_df)
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.sort_values(by='近1年', ascending=False)
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.reset_index(drop=True)
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.head(10)
    fund_em_open_fund_rank_df = fund_em_open_fund_rank_df.sort_values(by='近1年', ascending=True)
    def get_Top():
        fig = px.bar(fund_em_open_fund_rank_df, y='基金简称', x='近1年',
                     hover_data=['近1年'],
                     color='近1年', # 指定柱状图颜色根据 lifeExp字段数值大小自动着色
                     labels={'pop':'population of Canada'},
                     height=600, # 图表高度
                     width=800, # 图表宽度
                     title='公募基金涨幅Top10（近一年）',
                     orientation='h' # 条形图设置参数
                    )
        st.plotly_chart(fig)
    Top = get_Top()

elif choose == '基金经理':
    st.subheader('基金经理信息：')
    manager = pd.read_csv('fund_manager.csv')
    manager.sort_values("现任基金资产总规模", inplace=True, ascending=False)
    manager_guimo = manager.head(10)
    df = pd.read_csv("./fund_emotion/manager.csv")
    df = pd.DataFrame(df)

    # 管理规模
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



    # 最佳回报
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



    # 从业时间分布
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


    def get_duibi():
        fig = px.scatter_3d(df,
                            x='累计从业时间',
                            y='现任基金资产总规模',
                            z='现任基金最佳回报',
                            color='现任基金最佳回报',
                            size_max=18,
                            opacity=0.7,
                            title='从业时间(天)-管理资产规模(亿)-最佳回报(%)对比'
                            )

        st.plotly_chart(fig)


    manager1 = get_guimo()
    manager2 = get_huibao()
    manager3 = get_time()
    manager4 = get_duibi()

elif choose == '市场情绪':
    st.subheader('（易方达消费行业股票吧）市场情绪：')
    image2 = Image.open('img/img3.jpg')
    st.image(image2, use_column_width=True)
    comment = pd.read_csv('./fund_emotion/110022_guba_snownlp_nb_1.csv')
    comment = pd.DataFrame(comment)
    bl = comment['nb_result'].value_counts()
    comment['year'] = comment['time'].map(lambda x: x.split('/')[0])
    b2 = comment['year']
    comment_fl = pd.read_csv('./fund_emotion/comment_frequency.csv')
    comment_fl = pd.DataFrame(comment_fl)
    comment_f2 = pd.read_csv('./fund_emotion/110022_guba_snownlp_nb_2.csv')
    comment_f2 = pd.DataFrame(comment_f2)
    def get_bili():
        # 圆环图
        df = px.data.tips()
        fig = px.pie(
                    bl,
                    values='nb_result',
                    color_discrete_sequence=px.colors.sequential.Bluyl,
                    hole=0.6,  # 设置空心半径比例
                    title='市场情绪（积极/消极）',
                    )
        st.plotly_chart(fig)

    def get_bili1():
        fig = px.bar(comment_fl,
                     y='shuliang',
                     x='fenshu',
                     title='市场情绪评分分布情况：',
                     color='shuliang'
                         )
        st.plotly_chart(fig)


    def get_bili2():
        fig = px.bar(comment_f2,
                     x="month",
                     y=["good", "negative"],
                     title='市场情绪指数对比（月份）',
                     )
        st.plotly_chart(fig)

    def get_bili3():
        st.subheader('基金贴吧热词（Top-50）：')
        image = Image.open('./fund_emotion/data_merge_lstm/fund_cloud.jpg')
        st.image(image, caption='Sunrise by the mountains',
        use_column_width = True)

    comment1 = get_bili()
    comment2 = get_bili1()
    comment3 = get_bili2()
    comment4 = get_bili3()



