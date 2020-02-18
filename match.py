# -*- coding: utf-8 -*-
import pandas as pd
from tqdm import trange


def read_and_chop(file_name, sheet_name=0) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


def match(top_ten_df, detail_df):
    pe_ratio_ls = []
    type_ls = []

    for i in range(1, 11):
        col = f'第{i}大股东'
        ratio_col = f'第{i}大股东持股比例'
        temp = top_ten_df[[col, ratio_col]].reset_index().merge(
            detail_df[['股东', '性质']],
            left_on=col,
            right_on='股东',
        ).set_index('index')
        pe_ratio_ls.append(temp[ratio_col])
        type_ls.append(temp['性质'])

    top_ten_ratio = top_ten_df[[f'第{i}大股东持股比例' for i in range(1, 11)]].sum(axis=1)

    pe_ratio_df = pd.concat(pe_ratio_ls, axis=1)
    type_df = pd.concat(type_ls, axis=1)
    pe_num = pd.notna(pe_ratio_df).sum(axis=1).rename('PE_NUM')
    pe_ratio = pe_ratio_df.sum(axis=1).rename('PE_RATIO')
    state_num = (type_df == '国有').sum(axis=1).rename('STATE_NUM')
    state_ratio = ((type_df == '国有').values * pe_ratio_df).sum(axis=1).rename('STATE_RATIO')
    state_over_pe_num = (state_num / pe_num).rename('STATE_OVER_PE_NUM')
    state_over_pe_ratio = (state_ratio / pe_ratio).rename('STATE_OVER_PE_RATIO')

    result = pd.concat(
        objs=[
            pe_num,
            pe_ratio,
            state_num,
            state_ratio,
            state_over_pe_num,
            state_over_pe_ratio,
        ],
        axis=1,
    ).fillna(0)  # type: pd.DataFrame
    result['PE_OVER_TEN_RATIO'] = result['PE_RATIO'] / top_ten_ratio
    return result


# 去除最后两行
detail_df = read_and_chop('股东情况.xlsx', '股东')
detail_df = detail_df[
    (detail_df['类型'] == 'PE')
    | (detail_df['类型'] == 'VC')
    | (detail_df['类型'] == '战略投资者')
    ]

res_ls = []
input_io = pd.ExcelFile('股东情况.xlsx')
for i in trange(4):
    top_ten_df = read_and_chop(input_io, f't{i}前十大股东')
    res = match(top_ten_df, detail_df)[[
        'PE_NUM', 'PE_RATIO'
    ]]
    res = res.rename(columns={
        'PE_NUM': f'PE_NUM_{i}',
        'PE_RATIO': f'PE_RATIO_{i}',
    })
    res_ls.append(res)

result = pd.concat([top_ten_df.iloc[:, :5]] + res_ls, axis=1).fillna(0)
result.to_excel('股东情况匹配.xlsx')
# input_io.close()
