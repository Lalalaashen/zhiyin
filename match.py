# -*- coding: utf-8 -*-

import pandas as pd


def read_and_chop(file_name, sheet_name=0) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


# 去除最后两行
top_ten_df = read_and_chop('股东情况.xlsx', '前十大股东')
detail_df = read_and_chop('股东情况.xlsx', '股东')

detail_df = detail_df[
    (detail_df['类型'] == 'PE')
    | (detail_df['类型'] == 'VC')
    | (detail_df['类型'] == '战略投资者')
    ]

merged_df_ls = []
for i in range(1, 11):
    col = f'第{i}大股东'
    ratio_col = f'第{i}大股东持股比例'
    merged_df = top_ten_df[[col, ratio_col]].reset_index().merge(
        detail_df['股东'],
        left_on=col,
        right_on='股东',
    ).set_index('index')
    merged_df_ls.append(merged_df[ratio_col])

total_merged_df = pd.concat(merged_df_ls, axis=1)
result = pd.concat(
    objs=[
        top_ten_df.iloc[:, :5],
        pd.notna(total_merged_df).sum(axis=1).rename('数量'),
        total_merged_df.sum(axis=1).rename('总占比'),
    ],
    axis=1,
)  # type: pd.DataFrame
result.to_excel('股东情况匹配.xlsx')
