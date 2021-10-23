# 本文将使用sklearn实现朴素贝叶斯模型（基于snownlp/两者数据误差约为20%）
# 保存为110022_guba_snownlp_nb.csv
import pandas as pd
import numpy as np
from snownlp import SnowNLP
import jieba
import jieba

df = pd.read_csv("110022_guba_snownlp_all.csv")
df = pd.DataFrame(df)

# jieba分词
from sklearn.model_selection import train_test_split
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

df['cut_title'] = df.title.apply(chinese_word_cut)


from sklearn.feature_extraction.text import CountVectorizer
# 划分数据集
X = df['cut_title']
y = df.snlp_result

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=22)

# 词向量（数据处理）设置停用词表，这样的词我们就不会统计出来（多半是虚拟词，冠词等等）
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

stop_words_file = 'stopword.txt'
stopwords = get_custom_stopwords(stop_words_file)

vect = CountVectorizer(max_df = 0.8,
                       min_df = 3,
                       token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                       stop_words=frozenset(stopwords))

test = pd.DataFrame(vect.fit_transform(X_train).toarray(), columns=vect.get_feature_names())
# print(test.head())

# 训练模型
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

X_train_vect = vect.fit_transform(X_train)
nb.fit(X_train_vect, y_train)
train_score = nb.score(X_train_vect, y_train)
print(train_score)# result 0.8377734
# 测试数据
X_test_vect = vect.transform(X_test)
print(nb.score(X_test_vect, y_test))
# 数据存储
X_vec = vect.transform(X)
nb_result = nb.predict(X_vec)
df['nb_result'] = nb_result


df = df.drop([ 'snlp_result','cut_title'], axis=1)
df = df.to_csv('110022_guba_snownlp_nb.csv',encoding='utf_8_sig',index=False)
print(df)
# #定义拼接函数，并对字段进行去重
# def concat_func(x):
#     return pd.Series({
#         # 'nb_result':','.join('%s' % id for id in x['nb_result']),
#         'nb_result': ','.join('%s' % id for id in x['nb_result']),
#     }
#     )
#
# #分组聚合+拼接
# result=df.groupby(df['time']).apply(concat_func).reset_index()
# result = result.sort_values(by='time',ascending=False)
# result = result.reset_index(drop = True)

# print(result['nb_result'].value_counts())



# df['different'] = np.where(df['snlp_result']==df['nb_result'],'','no')
# no=df['different'].value_counts()
# all=len(df)
# a = no/all
# print(df)
# result = result.to_csv('110022_guba_snownlp_nb.csv',encoding='utf_8_sig',index=False)
