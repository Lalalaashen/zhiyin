# -*- coding: utf-8 -*-
# %% Init
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import OneHotEncoder

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def read_and_chop(file_name, sheet_name=0) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


def get_dummy(ind):
    """生成哑变量"""
    ohe = OneHotEncoder()
    ohe.fit(ind)
    return ohe.transform(ind).toarray()


# %% 读取
Xs = read_and_chop('重大重组事件V3_无st_尝试.xlsx', '长期绩效变量').fillna(0)
# Xs = read_and_chop('重大重组事件V4_无st.xlsx', '长期绩效变量').fillna(0)

# # %% 数据clip
controlled = ['RELATIVE SIZE', 'LN_ASSET', 'GROWTH', 'LEV', 'BROE', 'CASHPAY', 'OCF']
# clip
for c in controlled:
    Xs[c] = Xs[c].clip(*np.percentile(Xs[c], [1, 99]))

# %% 数据处理
# 去除行业数量太小的
ind_clip_num = 1
ind_value_counts = Xs['INDU'].value_counts()
others = ind_value_counts[ind_value_counts <= ind_clip_num].index
for other in others:
    Xs.loc[Xs['INDU'] == other, 'INDU'] = '其他'
# print(Xs['INDU'].value_counts())

# %% 生成哑变量
# 行业哑变量
ind_ohe_trans = get_dummy(Xs['INDU'].values.reshape((-1, 1)))[:, :-1]  # 以最后一个为基准
for i in range(ind_ohe_trans.shape[1]):
    Xs['IND_{}'.format(i)] = ind_ohe_trans[:, i]
# 年度哑变量
year_ohe_trans = get_dummy(Xs['YEAR'].values.reshape((-1, 1)))[:, :-1]  # 以最后一个为基准
for i in range(year_ohe_trans.shape[1]):
    Xs['YEAR_{}'.format(i)] = year_ohe_trans[:, i]

# %% 回归并打印结果
while True:
    # %% 读取X
    # to_add = [
    #     'PE_1',
    #     'PE_UNITED',
    #     # 'PE_RATIO',
    #     # 'STATE_NUM',
    #     # 'STATE_RATIO',
    #     # 'STATE_OVER_PE_NUM',
    #     'STATE_OVER_PE_RATIO',
    #     # 'PE_OVER_TEN_RATIO',
    # ]
    try:
        to_add = input('输入变量名，空格分开').split(' ')
        ind_dummy = ['IND_{}'.format(i) for i in range(ind_ohe_trans.shape[1])]
        year_dummy = ['YEAR_{}'.format(i) for i in range(ind_ohe_trans.shape[1])]
        xs_cols = controlled + to_add + ind_dummy + year_dummy
        xs = Xs[xs_cols]
        x = sm.add_constant(xs)  # 若模型中有截距
        cared = to_add
        cared_index = [xs_cols.index(x) + 1 for x in cared]  # 带常量的index

        y_cols = ['t0-t-1', 't1-t-1', 't2-t-1', 't3-t-1']
        df = pd.DataFrame(
            index=cared,
            columns=pd.MultiIndex.from_product([y_cols, ['pvalue', 'beta']])
        )
        print(cared)
        for col in y_cols:
            y = Xs[col]
            model = sm.OLS(y, x).fit()  # 构建最小二乘模型并拟合
            # print(model.summary())  # 输出所有回归结果
            df[(col, 'pvalue')] = model.pvalues[cared_index].values
            df[(col, 'beta')] = model.params[cared_index].values
        print(df)
    except:
        pass
