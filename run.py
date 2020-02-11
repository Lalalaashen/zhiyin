# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import math as math
import scipy.stats as st
import factor_analyzer


def load_data(io, sheet_name):
    """读取本地数据"""
    df = pd.read_excel(io, sheet_name=sheet_name).dropna()
    df['年份'] = df['报告期'].apply(lambda x: x.year)
    return df


def KMO_stat(df):
    """计算KMO统计量"""
    dataset_corr = df.corr().values
    corr_inv = np.linalg.inv(dataset_corr)
    nrow_inv_corr, ncol_inv_corr = dataset_corr.shape
    A = np.ones((nrow_inv_corr, ncol_inv_corr))
    for i in range(0, nrow_inv_corr, 1):
        for j in range(i, ncol_inv_corr, 1):
            A[i, j] = -(corr_inv[i, j]) / (math.sqrt(corr_inv[i, i] * corr_inv[j, j]))
            A[j, i] = A[i, j]
    dataset_corr = np.asarray(dataset_corr)
    kmo_num = np.sum(np.square(dataset_corr)) - np.sum(np.square(np.diagonal(A)))
    kmo_denom = kmo_num + np.sum(np.square(A)) - np.sum(np.square(np.diagonal(A)))
    kmo_value = kmo_num / kmo_denom
    return kmo_value


def bartlett_stat(df):
    """计算bartlett统计量，返回近似卡方，自由度，显著性"""
    corr = df.corr().values
    detCorr = np.linalg.det(corr)
    n = len(df)
    p = len(df.columns)
    statistic = -math.log(detCorr) * (n - 1 - (2 * p + 5) / 6)
    dof = p * (p - 1) / 2
    # 双侧概率
    pval = (1.0 - st.chi2.cdf(statistic, dof)) * 2
    return statistic, dof, pval


def pca_analysis(df, eigen_threshold=1):
    """主成分分析"""

    # 初始分析
    fa = factor_analyzer.FactorAnalyzer(n_factors=df.shape[1], method='principal', rotation=None)
    fa.fit(df)
    init_var = pd.DataFrame(
        data=fa.get_factor_variance(),
        index=pd.MultiIndex.from_product([["初始特征值"], ['总计', '方差占比', '累计方差占比']]),
        columns=range(1, df.shape[1] + 1),
    ).T

    # 选中因子再分析
    select_num = np.where(fa.get_eigenvalues()[0] > eigen_threshold)[0][-1] + 1
    fa = factor_analyzer.FactorAnalyzer(n_factors=select_num, method='principal', rotation=None)
    fa.fit(df)
    extract = pd.DataFrame(
        data=np.hstack([np.array([[1] for _ in range(df.shape[1])]), fa.get_communalities().reshape((-1, 1))]),
        index=df.columns,
        columns=pd.MultiIndex.from_product([["公因子方差"], ['初始', '提取']]),
    )
    print(extract)
    print(init_var)
    print('选取前{}个因子'.format(select_num))
    print('成分得分系数矩阵')
    score_corr = pd.DataFrame(
        data=np.linalg.solve(fa.corr_, fa.loadings_),
        columns=pd.MultiIndex.from_product([["成分得分系数矩阵"], ['Factor_{}'.format(x) for x in range(1, select_num + 1)]]),
        index=df.columns,
    )
    print(score_corr)
    factor = pd.DataFrame(
        data=fa.transform(df),
        columns=['Factor_{}'.format(x) for x in range(1, select_num + 1)],
    )

    # 旋转
    fa = factor_analyzer.FactorAnalyzer(n_factors=select_num, method='principal', rotation='varimax')
    fa.fit(df)
    rotate_score_corr = pd.DataFrame(
        data=np.linalg.solve(fa.corr_, fa.loadings_),
        columns=pd.MultiIndex.from_product([["旋转成分得分系数矩阵"], ['Factor_{}'.format(x) for x in range(1, select_num + 1)]]),
        index=df.columns,
    )
    print(rotate_score_corr)
    rotate_var = pd.DataFrame(
        data=fa.get_factor_variance(),
        index=pd.MultiIndex.from_product([["旋转特征值"], ['总计', '方差占比', '累计方差占比']]),
        columns=range(1, select_num + 1),
    ).T
    factor['综合得分'] = np.dot(factor.values, rotate_var[('旋转特征值', '方差占比')].values.reshape(-1, 1))
    return init_var, rotate_var, select_num, extract, score_corr, rotate_score_corr, factor


if __name__ == '__main__':
    # -----------------参数-----------------
    input_name = '重大重组事件V3_无st.xlsx'  # 读取的文件名称
    output_name = '重大重组事件结果.xlsx'  # 读取的文件名称
    cols = [
        '净资产收益率',
        '总资产报酬率',
        '每股收益',
        '流动比率',
        '速动比率',
        '总资产周转率',
        '流动资产周转率',
        '总资产增长率',
        '营业收入增长率',
        '每股经营活动净现金流',
    ]  # 要读取的变量名称
    eigen_threshold = 1  # 主成分分析特征值的筛选阈值
    idx_ls = [-1, 0, 1, 2, 3]

    # -----------------运行-----------------
    input_io = pd.ExcelFile(input_name)
    df_ls = [load_data(input_io, '财务指标t{}'.format(idx)) for idx in idx_ls]

    output_io = pd.ExcelWriter(output_name)
    # 计算描述性统计量
    start_row = 0
    for i, (idx, df) in enumerate(zip(idx_ls, df_ls)):
        print('#' * 20, 't =', idx, '描述性统计量', '#' * 20)
        describe = df[cols].describe().T
        describe.columns = pd.MultiIndex.from_product([["财务指标t{}".format(idx)], describe.columns])
        print(describe)
        print()
        describe.to_excel(output_io, '描述性统计量'.format(idx), startrow=start_row)
        start_row += describe.shape[0] + 4
    # 计算KMO统计量
    print('#' * 20, 'KMO_stat', '#' * 20)
    kmo = pd.DataFrame(
        data=np.array([KMO_stat(df[cols]) for df in df_ls]),
        index=idx_ls,
        columns=pd.MultiIndex.from_product([["KMO_stat"], ['KMO_stat']]),
    )
    print(kmo)
    print()
    kmo.to_excel(output_io, 'kmo_bartlett', startcol=0)
    # 计算bartlett统计量
    print('#' * 20, 'bartlett_stat', '#' * 20)
    bartlett = pd.DataFrame(
        data=np.array([bartlett_stat(df[cols]) for df in df_ls]),
        index=idx_ls,
        columns=pd.MultiIndex.from_product([["bartlett"], ['stat', 'dof', 'pval']]),
    )
    print(bartlett)
    print()
    bartlett.to_excel(output_io, 'kmo_bartlett', startcol=2)
    # 主成分分析
    score_ls = []
    for idx, df in zip(idx_ls, df_ls):
        print('#' * 20, 't =', idx, '主成分分析', '#' * 20)
        sheet_name = '财务指标t{}'.format(idx)
        init_var, rotate_var, select_num, extract, \
        score_corr, rotate_score_corr, factor = pca_analysis(df[cols], eigen_threshold)
        # 写入
        extract.to_excel(output_io, sheet_name, startrow=0, startcol=0)
        init_var.to_excel(output_io, sheet_name, startrow=0, startcol=4)
        init_var.iloc[:select_num].rename(columns={'初始特征值': '提取载荷平方和'}).to_excel(
            output_io, sheet_name, startrow=0, startcol=8,
        )
        rotate_var.rename(columns={'旋转特征值': '提取旋转载荷平方和'}).to_excel(
            output_io, sheet_name, startrow=0, startcol=12,
        )
        score_corr.to_excel(output_io, sheet_name, startrow=len(cols) + 5, startcol=0)
        rotate_score_corr.to_excel(output_io, sheet_name, startrow=len(cols) + 5, startcol=7)
        factor.to_excel(output_io, sheet_name, startrow=2 * len(cols) + 9, startcol=0)
        score_ls.append(factor['综合得分'])
    # 去除市场
    to_calc = pd.concat([
        pd.concat([df['年份'] for df in df_ls], axis=0),
        pd.concat(score_ls, axis=0),
    ], axis=1,
    )  # type: pd.DataFrame
    year_mean = to_calc.groupby(['年份']).mean()
    year_mean.to_excel(output_io, '得分年均数据', startrow=0, startcol=0)
    to_calc = to_calc.join(year_mean, on=['年份'], lsuffix='_原始', rsuffix='_年均')
    to_calc['修正得分'] = to_calc['综合得分_原始'] - to_calc['综合得分_年均']

    for i in range(len(df_ls)):
        to_calc.iloc[i * df.shape[0]:(i + 1) * df.shape[0]].to_excel(
            output_io, '财务指标t{}'.format(idx_ls[i]), startrow=2 * len(cols) + 9, startcol=6,
        )

    score_ls = [to_calc['修正得分'].iloc[i * df.shape[0]:(i + 1) * df.shape[0]] for i in range(len(df_ls))]
    # t检验
    diff_score = pd.concat([
        score_ls[i + 1] - score_ls[i] for i in range(0, len(idx_ls) - 1)
    ], axis=1)
    diff_score.columns = ['t{}-t{}'.format(idx_ls[i + 1], idx_ls[i]) for i in range(0, len(idx_ls) - 1)]
    t_stat_df = pd.DataFrame(
        data=np.vstack([
            diff_score.mean().values,
            np.vstack([st.ttest_rel(score_ls[i + 1], score_ls[i]) for i in range(4)]).T,
        ]),
        columns=diff_score.columns,
        index=['mean', 'statistic', 'pvalue']
    )
    print(t_stat_df)
    diff_score.to_excel(output_io, '得分差分t检验', startrow=0, startcol=0)
    t_stat_df.to_excel(output_io, '得分差分t检验', startrow=0, startcol=diff_score.shape[1] + 3)
    # 储存
    output_io.save()
