# -*- coding: utf-8 -*-
import os, requests
import pandas as pd
from lxml import etree

# TYPE = "科创板"
TYPE = "创业板"
DIR_PATH = "./招股说明书/"
FILE_DIR_PATH = f"./招股说明书/{TYPE}招股说明书/"


def download():
    df0 = pd.read_excel(os.path.join(DIR_PATH, f"{TYPE}招股说明书.xlsx"))
    code_set = set()
    for i in range(0, len(df0)):
        ori_str = df0["链接"][i]
        name_document = df0["证券代码"][i]
        file_name = FILE_DIR_PATH + name_document + ".pdf"
        if os.path.exists(file_name):
            code_set.add(name_document)
            continue
        try:
            beginplace = 0
            endplace = 0
            for j in range(0, len(ori_str)):
                if ori_str[j] == "\"" and beginplace != 0:
                    endplace = j
                    break
                if ori_str[j] == "\"" and beginplace == 0:
                    beginplace = j + 1

            url = ori_str[beginplace:endplace]
            # url="http://news.windin.com/ns/bulletin.php?code=6B166667E7D7&id=97611014&type=1"
            req = requests.get(url)
            wb_data = req.text
            html = etree.HTML(wb_data)
            html_data = html.xpath('//*[@id="mdiv"]/div[3]/div[2]/a')
            x = html_data[0].get("href")
            dldsite = "http://news.windin.com/ns/" + x
            r2 = requests.get(dldsite)

            with open(file_name, "wb") as code:
                code.write(r2.content)
            print(name_document)
            code_set.add(name_document)
        except:
            pass
    return code_set


if __name__ == '__main__':
    downloaded_set = download()
    total_set = set(pd.read_excel(os.path.join(DIR_PATH, f"{TYPE}.xlsx"))['代码'])
    print('未下载的招股说明书')
    print(sorted(total_set - downloaded_set))
