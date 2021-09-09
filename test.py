import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
('基金查询','基金经理', '基金排行'))
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

    fund = fund.to_csv('fund.csv', index=None)
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

    fileTrain = './data/accTrain.csv'
    jjTrain = [id]
    fileTest = './data/accTest.csv'
    jjTest = id


    def getUrl(fscode):
        head = 'http://fund.eastmoney.com/pingzhongdata/'
        tail = '.js?v=' + time.strftime("%Y%m%d%H%M%S", time.localtime())
        return head + fscode + tail


    # 根据基金代码获取净值
    def getWorth(fscode):
        content = requests.get(getUrl(fscode))
        jsContent = execjs.compile(content.text)
        # 单位净值走势
        netWorthTrend = jsContent.eval('Data_netWorthTrend')
        # 累计净值走势
        ACWorthTrend = jsContent.eval('Data_ACWorthTrend')
        netWorth = []
        ACWorth = []
        for dayWorth in netWorthTrend[::-1]:
            netWorth.append(dayWorth['y'])
        for dayACWorth in ACWorthTrend[::-1]:
            ACWorth.append(dayACWorth[1])
        return netWorth, ACWorth


    ACWorthFile = open(fileTrain, 'w')
    for code in jjTrain:
        try:
            _, ACWorth = getWorth(code)
        except:
            continue
        if len(ACWorth) > 0:
            ACWorthFile.write(",".join(list(map(str, ACWorth))))
            ACWorthFile.write("\n")
            print('{} data downloaded1'.format(code))
    ACWorthFile.close()

    ACWorthTestFile = open(fileTest, 'w')
    _, ACWorth = getWorth(jjTest)
    if len(ACWorth) > 0:
        ACWorthTestFile.write(",".join(list(map(str, ACWorth))))
        ACWorthTestFile.write("\n")
        print('{} data downloaded'.format(jjTest))
    ACWorthTestFile.close()

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
    # 预测
    import numpy as np
    import pandas as pd
    import csv
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import LSTM
    from matplotlib import pyplot as plt
    import plotly.figure_factory as ff
    import plotly_express as px
    import streamlit as st

    plt.rcParams['font.sans-serif'] = 'SimHei'
    plt.rcParams['axes.unicode_minus'] = False

    batch_size = 4
    epochs = 50
    time_step = 4  # 用多少组天数进行预测
    input_size = 3  # 每组天数，亦即预测天数
    look_back = time_step * input_size
    showdays = 120  # 最后画图观察的天数（测试天数）

    X_train = []
    y_train = []
    X_validation = []
    y_validation = []
    testset = []  # 用来保存测试基金的近期净值

    # 忽略掉最近的forget_days天数据（回退天数，用于预测的复盘）
    forget_days = 1


    def create_dataset(dataset):
        dataX, dataY = [], []
        print('len of dataset: {}'.format(len(dataset)))
        for i in range(0, len(dataset) - look_back, input_size):
            x = dataset[i: i + look_back]
            dataX.append(x)
            y = dataset[i + look_back: i + look_back + input_size]
            dataY.append(y)
        return np.array(dataX), np.array(dataY)


    def build_model():
        model = Sequential()
        model.add(LSTM(units=128, input_shape=(time_step, input_size)))
        model.add(Dense(units=input_size))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model


    # 设定随机数种子
    seed = 7
    np.random.seed(seed)

    my_bar.progress(20)

    # 导入数据（训练集）
    with open(fileTrain) as f:
        row = csv.reader(f, delimiter=',')
        for r in row:
            dataset = []
            r = [x for x in r if x != 'None']
            # 涨跌幅是2天之间比较，数据会减少1个
            days = len(r) - 1
            # 有效天数太少，忽略
            if days <= look_back + input_size:
                continue
            for i in range(days):
                f1 = float(r[i])
                f2 = float(r[i + 1])
                if f1 == 0 or f2 == 0:
                    dataset = []
                    break
                # 把数据放大100倍，相当于以百分比为单位
                f2 = (f2 - f1) / f1 * 100
                # 如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
                if f2 > 15 or f2 < -15:
                    dataset = []
                    break
                dataset.append(f2)
            n = len(dataset)
            # 进行预测的复盘，忽略掉最近forget_days的训练数据
            n -= forget_days
            if n >= look_back + input_size:
                # 如果数据不是input_size的整数倍，忽略掉最前面多出来的
                m = n % input_size
                X_1, y_1 = create_dataset(dataset[m:n])
                X_train = np.append(X_train, X_1)
                y_train = np.append(y_train, y_1)

    my_bar.progress(30)

    # 导入数据（测试集）
    with open(fileTest) as f:
        row = csv.reader(f, delimiter=',')
        # 写成了循环，但实际只有1条测试数据
        for r in row:
            dataset = []
            # 去掉记录为None的数据（当天数据缺失）
            r = [x for x in r if x != 'None']
            # 涨跌幅是2天之间比较，数据会减少1个
            days = len(r) - 1
            # 有效天数太少，忽略，注意：测试集最后会虚构一个input_size
            if days <= look_back:
                print('only {} days data. exit.'.format(days))
                continue
            # 只需要最后画图观察天数的数据
            if days > showdays:
                r = r[days - showdays:]
                days = len(r) - 1
            for i in range(days):
                f1 = float(r[i])
                f2 = float(r[i + 1])
                if f1 == 0 or f2 == 0:
                    print('zero value found. exit.')
                    dataset = []
                    break
                # 把数据放大100倍，相当于以百分比为单位
                f2 = (f2 - f1) / f1 * 100
                # 如果涨跌幅绝对值超过15%，基金数据恐有问题，忽略该组数据
                if f2 > 15 or f2 < -15:
                    print('{} greater then 15 percent. exit.'.format(f2))
                    dataset = []
                    break
                testset.append(f1)
                dataset.append(f2)
            # 保存最近一天基金净值
            f1 = float(r[days])
            testset.append(f1)
            # 测试集虚构一个input_size的数据（若有forget_days的数据，则保留）
            if forget_days < input_size:
                for i in range(forget_days, input_size):
                    dataset.append(0)
                    testset.append(np.nan)
            else:
                dataset = dataset[:len(dataset) - forget_days + input_size]
                testset = testset[:len(testset) - forget_days + input_size]
            if len(dataset) >= look_back + input_size:
                # 将testset修正为input_size整数倍加1
                m = (len(testset) - 1) % input_size
                testset = testset[m:]
                m = len(dataset) % input_size
                # 将dataset修正为input_size整数倍
                X_validation, y_validation = create_dataset(dataset[m:])

    # 将输入转化成[样本数，时间步长，特征数]
    X_train = X_train.reshape(-1, time_step, input_size)
    X_validation = X_validation.reshape(-1, time_step, input_size)

    # 将输出转化成[样本数，特征数]
    y_train = y_train.reshape(-1, input_size)
    y_validation = y_validation.reshape(-1, input_size)

    print('num of X_train: {}\tnum of y_train: {}'.format(len(X_train), len(y_train)))
    print('num of X_validation: {}\tnum of y_validation: {}'.format(len(X_validation), len(y_validation)))

    my_bar.progress(40)

    # 训练模型
    model = build_model()
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, validation_split=0.25, shuffle=True)

    my_bar.progress(60)

    # 评估模型
    train_score = model.evaluate(X_train, y_train, verbose=0)
    validation_score = model.evaluate(X_validation, y_validation, verbose=0)

    # 预测
    predict_validation = model.predict(X_validation)

    my_bar.progress(80)

    # 将之前虚构的最后一组input_size里面的0涨跌改为NAN（不显示虚假的0）
    if forget_days < input_size:
        for i in range(forget_days, input_size):
            y_validation[-1, i] = np.nan

    print('Train Set Score: {:.3f}'.format(train_score))
    print('Test Set Score: {:.3f}'.format(validation_score))
    print('未来{}天实际百分比涨幅为：{}'.format(input_size, y_validation[-1]))
    print('未来{}天预测百分比涨幅为：{}'.format(input_size, predict_validation[-1]))

    # 进行reshape(-1, 1)是为了plt显示
    y_validation = y_validation.reshape(-1, 1)
    predict_validation = predict_validation.reshape(-1, 1)
    testset = np.array(testset).reshape(-1, 1)

    # 图表显示
    fig = plt.figure(figsize=(15, 6))
    plt.plot(y_validation, color='blue', label='基金每日涨幅')
    plt.plot(predict_validation, color='red', label='预测每日涨幅')
    plt.legend(loc='upper left')
    plt.title('关联组数：{}组，预测天数：{}天，回退天数：{}天'.format(time_step, input_size, forget_days))
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


elif choose == '基金经理':
    st.subheader('基金经理信息：')
    manager = pd.read_csv('fund_manager.csv')
    manager.sort_values("现任基金资产总规模", inplace=True, ascending=False)
    manager_guimo = manager.head(10)

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

    manager1 = get_guimo()
    manager2 = get_huibao()
    manager3 = get_time()




