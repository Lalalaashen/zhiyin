# -*- coding: utf-8 -*-
# %% 配置
import pickle
import pandas as pd
import numpy as np
from tqdm import trange, tqdm
from WindPy import w


def read_and_chop(file_name, sheet_name, **kwargs) -> pd.DataFrame:
    """读取并截取有效数据"""
    df = pd.read_excel(file_name, sheet_name=sheet_name, **kwargs)
    if df.iloc[-1, 0] == '数据来源：Wind':
        df = df.iloc[:-2]
    return df


w.start()

# %% 读取
file_name = '股东情况.xlsx'
print('reading from', file_name)
input_io = pd.ExcelFile(file_name)
# 事件
stock_info = read_and_chop(input_io, 't0前十大股东')
stock_info['报告期'] = stock_info['报告期'].apply(lambda x: x.strftime(format='%Y%m%d'))
# PE
pe_df = read_and_chop(input_io, '股东')
pe_df = pe_df[
    (pe_df['类型'] == 'PE')
    | (pe_df['类型'] == 'VC')
    | (pe_df['类型'] == '战略投资者')
    ]
pe_set = set(pe_df['股东'])

# %% 从WIND拉取数据
md_ls = ['0331', '0630', '0930', '1231']
codes = []
report_dates = []
ipo_dates = []
target_dates_ls = []
holder_names_ls = []


def get_index(month):
    idx = month // 3
    if month % 3 == 0:
        idx -= 1
    return idx


for code, report_date in tqdm(zip(stock_info['股票代码'].to_list(), stock_info['报告期'].to_list()),
                              total=stock_info.shape[0]):
    # print(code, date)
    res = w.wss(code, "holder_name,ipo_date", f"tradeDate={report_date};order=0")
    holder_name = res.Data[0][0].split(';')
    ipo_date = res.Data[1][0]
    holder_names = []
    target_dates = []
    if any(h in pe_set for h in holder_name):
        holder_names.append(holder_name)
        target_dates.append(report_date)
        dates = []
        # 获取dates列表
        for year in np.arange(ipo_date.year, int(report_date[:4]) + 1):
            if year == ipo_date.year:
                start_idx = get_index(ipo_date.month)
            else:
                start_idx = 0
            for md in md_ls[start_idx:]:
                # print(f'{year}{md}')
                dates.append(f'{year}{md}')
        # 存储
        assert dates[-1] == report_date
        for target_date in dates[-2::-1]:
            res = w.wss(code, "holder_name", f"tradeDate={target_date};order=0")
            if any(h in pe_set for h in res.Data[0][0].split(';')):
                # print(code, target_date, res.Data[0][0])
                target_dates.append(target_date)
                holder_names.append(res.Data[0][0].split(';'))
            # else:
            #     break
    codes.append(code)
    report_dates.append(report_date)
    ipo_dates.append(f'{ipo_date.year}{md_ls[get_index(ipo_date.month)]}')
    target_dates_ls.append(target_dates)
    holder_names_ls.append(holder_names)

# %% 分析
in_out_dict = {}
for code, report_date, ipo_date, target_dates, holder_names in zip(
        codes, report_dates, ipo_dates, target_dates_ls, holder_names_ls
):
    content_dict = {}
    if len(holder_names) > 0:
        for target_date, holder_name in zip(target_dates, holder_names):
            for pe_holder_name in [x for x in holder_name if x in pe_set]:
                if pe_holder_name not in content_dict:
                    content_dict[pe_holder_name] = []
                content_dict[pe_holder_name].append(target_date)  # 存续期
    for pe in content_dict.keys():
        content_dict[pe] = sorted(content_dict[pe])
        print(f'{(code, report_date)}-{ipo_date}-{pe}-{"-".join(content_dict[pe])}')
    in_out_dict[(code, report_date)] = content_dict
# 存储
with open('in_out_dict.pkl', 'wb') as f:
    pickle.dump(in_out_dict, f)
# %% 拼合
df = pd.DataFrame(
    index=stock_info.index,
    columns=list(stock_info.columns[:5]) + [
        'IPO_DATE', 'HIST_PE_NUM', 'B_IPO_NUM',
    ] + [f'{x}_{i}' for i in range(1, 11) for x in ['PE', 'START', 'END', 'B_IPO']]
)
df.iloc[:, :5] = stock_info.iloc[:, :5]
# 匹配
for i, (code, report_date) in enumerate(zip(stock_info['股票代码'].to_list(), stock_info['报告期'].to_list())):
    df.loc[i, 'IPO_DATE'] = ipo_dates[i]
    if (code, report_date) in in_out_dict:
        content_dict = in_out_dict[(code, report_date)]
        df.loc[i, 'HIST_PE_NUM'] = len(content_dict)
        for j, (key, value) in enumerate(content_dict.items()):
            df.loc[i, f'PE_{j + 1}'] = key
            df.loc[i, f'START_{j + 1}'] = value[0]
            df.loc[i, f'END_{j + 1}'] = value[-1]
            # 一直前十大股东的PE数量
            df.loc[i, f'B_IPO_{j + 1}'] = int(value[0] == ipo_dates[i] and value[-1] == report_date)
df['B_IPO_NUM'] = df[[f'B_IPO_{i}' for i in range(1, 11)]].sum(axis=1)
df = df.dropna(axis=1, how='all')
df.to_excel('私募存续期数据.xlsx', index=False)
# %% 结束
w.stop()

# %% 添加列
df = pd.read_excel('私募存续期数据.xlsx')
df['首次披露日期'] = df['首次披露日期'].apply(lambda x: int(x.strftime('%Y%m%d')))
# 读取
with open('in_out_dict.pkl', 'rb') as f:
    in_out_dict = pickle.load(f)
# 计算最早进入时间
earliest_entry = '并购当年PE最早进入时间'
mean_year_len = '并购当年PE平均持股年限'
to_add_cols = [earliest_entry, mean_year_len]
for col in to_add_cols:
    df[col] = None

for i in trange(stock_info.shape[0]):
    code = stock_info.loc[i, '股票代码']
    report_date = stock_info.loc[i, '报告期']
    if (code, report_date) in in_out_dict:
        temp_dict = in_out_dict[(code, report_date)]
        holders = stock_info.loc[i, [f'第{j}大股东' for j in range(1, 11)]]
        start_years = np.array([int(temp_dict[x][0]) if x in temp_dict else np.nan for x in holders])
        year_lens = np.array(
            [(df.loc[i, '最新披露日期'] - pd.to_datetime(temp_dict[x][0])).days if x in temp_dict else np.nan for x in holders]
        ) / 365.0
        mask = start_years > df.loc[i, '首次披露日期']
        start_years[mask] = np.nan
        year_lens[mask] = np.nan
        df.loc[i, earliest_entry] = np.nanmin(start_years)
        df.loc[i, mean_year_len] = np.nanmean(year_lens)

for col in ['报告期', 'IPO_DATE',
            earliest_entry] + [x for x in df.columns if x.startswith('START') or x.startswith('END')]:
    df[col] = df[col].apply(lambda x: str(int(x)) if not np.isnan(x) else x)
    df[col] = df[col].apply(lambda x: f'{x[:4]}/{x[4:6]}/{x[6:]}' if isinstance(x, str) else x)
df['PE_TIME'] = (df['最新披露日期'] - pd.to_datetime(df[earliest_entry])).apply(lambda x: x.days) / 365.
new_cols = list(df.columns)
for _ in to_add_cols + ['PE_TIME']:
    new_cols.insert(5, new_cols.pop(-1))
df = df[new_cols]
df.to_excel('私募存续期数据_1.xlsx', index=False)
