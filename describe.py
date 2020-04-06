# -*- coding: utf-8 -*-
# %% 配置
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import scipy.stats as stats
import statsmodels.api as sm
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor


# %% 读取数据
def read_and_chop(file_name, sheet_name, **kwargs) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name, **kwargs)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


df = read_and_chop('样本数据.xlsx', 'V1')
# 输出的IO
output_name = '样本数据统计分析.xlsx'
output_io = pd.ExcelWriter(output_name)

# %% 分行业情况
# 行业数为1的记为其他
indu_count_df = df.groupby(by='INDU')['INDU'].count()
for indu in indu_count_df.index[indu_count_df <= 1]:
    df.loc[df['INDU'] == indu, 'INDU'] = '其他'
# 全样本的分行业情况
indu_count_df = df.groupby(by='INDU')['INDU'].count()
indu_count_df.to_excel(output_io, '分行业情况', startrow=0, startcol=0, index_label='全样本')
# PE_YES = 1的分行业情况
indu_pe_count_df = df[df['PE_YES'] == 1].groupby(by='INDU')['INDU'].count()
indu_pe_count_df.to_excel(output_io, '分行业情况', startrow=indu_count_df.shape[0] + 2, startcol=0, index_label='PE_YES = 1')

# %% 描述性统计
# 连续变量CLIP
clip_cols = [
    'RELATIVE SIZE',
    'LN_ASSET',
    'LEV',
    'BROE',
    'GROWTH',
    'FCFF',
    'BLOCK',
    'PE_RATIO',
    'PE_TIME',
    'STATE_OVER_PE_RATIO',
]
start_row = 0
df[clip_cols] = df[clip_cols].clip(*np.percentile(df[clip_cols], [1, 99], axis=0), axis=1)
des_df = df.describe().T
des_df = des_df.drop(index='YEAR')
des_df.to_excel(output_io, '描述性统计', startrow=start_row, startcol=0, index_label='全样本')
start_row += des_df.shape[0] + 2
# PE_YES = 1
pe_yes_df = df[df['PE_YES'] == 1]
des_pe_yes_df = pe_yes_df.describe().T
des_pe_yes_df = des_pe_yes_df.drop(index='YEAR')
des_pe_yes_df.to_excel(output_io, '描述性统计', startrow=start_row, startcol=0, index_label='PE_YES = 1')
start_row += des_pe_yes_df.shape[0] + 2
# PE_YES = 0
no_pe_yes_df = df[df['PE_YES'] == 0]
des_no_pe_yes_df = no_pe_yes_df.describe().T
des_no_pe_yes_df = des_no_pe_yes_df.drop(index='YEAR')
des_no_pe_yes_df.to_excel(output_io, '描述性统计', startrow=start_row, startcol=0, index_label='PE_YES = 0')
start_row += des_no_pe_yes_df.shape[0] + 2
# t检验
t_test = pd.DataFrame(
    index=des_df.index,
    columns=pd.MultiIndex.from_product([['PE_YES = 1', 'PE_YES = 0'], ['nobs', 'mean', 'std']]).append(
        pd.MultiIndex.from_product([['t_test'], ['stat', 'p_value']])
    ),
)
t_test[('PE_YES = 1', 'nobs')] = des_pe_yes_df['count']
t_test[('PE_YES = 1', 'mean')] = des_pe_yes_df['mean']
t_test[('PE_YES = 1', 'std')] = des_pe_yes_df['std']
t_test[('PE_YES = 0', 'nobs')] = des_no_pe_yes_df['count']
t_test[('PE_YES = 0', 'mean')] = des_no_pe_yes_df['mean']
t_test[('PE_YES = 0', 'std')] = des_no_pe_yes_df['std']
t_test[('t_test', 'stat')], t_test[('t_test', 'p_value')] = st.ttest_ind_from_stats(
    mean1=t_test[('PE_YES = 1', 'mean')],
    std1=t_test[('PE_YES = 1', 'std')],
    nobs1=t_test[('PE_YES = 1', 'nobs')],
    mean2=t_test[('PE_YES = 0', 'mean')],
    std2=t_test[('PE_YES = 0', 'std')],
    nobs2=t_test[('PE_YES = 0', 'nobs')],
    equal_var=True,
)
t_test.to_excel(output_io, '描述性统计', startrow=start_row, startcol=0, index_label='t_test')


# %% 相关性分析
def get_stars(p_value):
    """***、**、*分别表示在 1%、5%、10%的水平上统计显著"""
    return '***' if p_value < 0.01 else '**' if p_value < 0.05 else '*' if p_value < 0.10 else ''


corr_cols = [
    'CAR1',
    'CAR3',
    'CAR5',
    't1-t-1',
    't2-t-1',
    't3-t-1',
    'PE_YES',
    'PE_TIME',
    'PE_UNITED',
    'PE_REP',
    'STATE_OVER_PE_RATIO',
    'RELATIVE SIZE',
    'CASHPAY',
    'LN_ASSET',
    'LEV',
    'BROE',
    'GROWTH',
    'FCFF',
    'BLOCK',
    'STATE',
]
corr_star_df = pd.DataFrame(index=corr_cols, columns=corr_cols)
for i in range(len(corr_cols)):
    for j in range(i + 1):
        col_1 = corr_cols[i]
        col_2 = corr_cols[j]
        corr, p_value = stats.pearsonr(df[col_1], df[col_2])
        stars = '' if i == j else get_stars(p_value)
        corr_star_df.loc[col_1, col_2] = f'{corr:.3f}{stars}'
corr_star_df.to_excel(output_io, '相关性分析', startrow=0, startcol=0, index_label='相关系数')


def plot_corr_matrix(corr_df, cmap='GnBu'):
    """绘制xs相关系数矩阵热力图"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    sns.heatmap(data=corr_df, ax=ax, cmap=cmap, square=True)  # type: plt.Axes
    ax.set_title('相关系数热力图')
    fig.tight_layout()
    fig.savefig('相关系数热力图.png', dpi=300, quality=100)


corr_df = df[corr_cols].corr()
rename_dict = {
    't1-t-1': 'F1-F-1',
    't2-t-1': 'F2-F-1',
    't3-t-1': 'F2-F-1',
}
corr_df = corr_df.rename(columns=rename_dict, index=rename_dict)
plot_corr_matrix(corr_df, cmap='Blues')


# %% 添加哑变量
def get_dummy(ind):
    """生成哑变量，以最后一个为基准"""
    ohe = OneHotEncoder(categories='auto')
    ohe.fit(ind)
    return ohe.transform(ind).toarray()[:, :-1]


def creat_df_with_dummy(df):
    """创建带有dummy的数据"""
    new_df = df.copy()
    # 行业哑变量
    ind_ohe_trans = get_dummy(new_df['INDU'].values.reshape((-1, 1)))
    for i in range(ind_ohe_trans.shape[1]):
        new_df['IND_{}'.format(i)] = ind_ohe_trans[:, i]
    # 年度哑变量
    year_ohe_trans = get_dummy(new_df['YEAR'].values.reshape((-1, 1)))
    for i in range(year_ohe_trans.shape[1]):
        new_df['YEAR_{}'.format(i)] = year_ohe_trans[:, i]
    return (
        new_df,
        ['IND_{}'.format(i) for i in range(ind_ohe_trans.shape[1])],
        ['YEAR_{}'.format(i) for i in range(year_ohe_trans.shape[1])],
    )


# 创建哑变量
df, ind_dummy, year_dummy = creat_df_with_dummy(df)
df_mask, ind_dummy_mask, year_dummy_mask = creat_df_with_dummy(df[df['PE_YES'] == 1].reset_index())


# %% 异方差检验
def het_test(y, xs):
    """回归"""
    # 构建最小二乘模型并拟合
    xs = sm.add_constant(xs)  # 截距
    model = sm.OLS(y, xs)
    res = model.fit()
    # White`s Test
    White_res = sm.stats.diagnostic.het_white(res.resid, model.exog)
    # BP Test
    BP_res = sm.stats.diagnostic.het_breuschpagan(res.resid, model.exog)
    return np.array(White_res), np.array(BP_res)


ctrl = [
    'RELATIVE SIZE', 'CASHPAY', 'LN_ASSET',
    'LEV', 'BROE', 'GROWTH',
    'FCFF', 'BLOCK', 'STATE',
]
char = [
    'PE_TIME', 'PE_UNITED',
    'PE_REP', 'STATE_OVER_PE_RATIO'
]
y_cols = ['t1-t-1', 't2-t-1', 't3-t-1', 'CAR1', 'CAR3', 'CAR5']
x_cols = ['PE_YES'] + ctrl + ind_dummy + year_dummy
x_cols_mask = char + ctrl + ind_dummy_mask + year_dummy_mask
het_test_df = pd.DataFrame(
    columns=y_cols,
    index=pd.MultiIndex.from_product([['模型1', '模型2'], ['White', 'BP'], ['lm', 'lm_p_value', 'f_value', 'f_p_value']])
)
for col in y_cols:
    # 第一个模型
    het_test_df.loc[('模型1', 'White'), col], \
    het_test_df.loc[('模型1', 'BP'), col] = het_test(y=df[col], xs=df[x_cols])
    # 第二个模型
    het_test_df.loc[('模型2', 'White'), col], \
    het_test_df.loc[('模型2', 'BP'), col] = het_test(y=df_mask[col], xs=df_mask[x_cols_mask])
het_test_df.to_excel(output_io, '异方差检验', startrow=0, startcol=0, index_label='异方差检验')

# %% 多重共线性检验
vif_x_cols_1 = ['CONS', 'PE_YES'] + ctrl
vif_x_cols_2 = ['CONS'] + char + ctrl
multi_test_df = pd.DataFrame(
    index=['CONS', 'PE_YES'] + char + ctrl,
    columns=['模型1', '模型2'],
)
df['CONS'] = 1
df_mask['CONS'] = 1
multi_test_df.loc[vif_x_cols_1, '模型1'] = np.array([
    variance_inflation_factor(df[vif_x_cols_1].values, i) for i in range(len(vif_x_cols_1))
])
multi_test_df.loc[vif_x_cols_2, '模型2'] = np.array([
    variance_inflation_factor(df_mask[vif_x_cols_2].values, i) for i in range(len(vif_x_cols_2))
])
multi_test_df.name = 'VIF'
multi_test_df.to_excel(output_io, '多重共线性检验', startrow=0, startcol=0, index_label='多重共线性检验')

# %% 储存
output_io.save()
