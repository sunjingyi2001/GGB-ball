import pandas as pd
import numpy as np
import datetime
import copy
import os
import openpyxl
import pickle
import math

def writeToExcel(file_path, new_list, title):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = title
    ws.cell(1, 1).value = str(new_list)
    # excel中的行和列是从1开始计数的，所以需要+1
    wb.save(file_path)  # 注意，写入后一定要保存
    print("成功写入文件: " + file_path + " !")
    return 1

def complete_data(data_name, real_att):
    obj_number, att_number = data_name.shape
    new_data = copy.deepcopy(data_name)

    # 处理分类属性（有序和无序）
    for k1 in range(att_number - real_att - 1):
        Va_not_mising = []
        Va_mising_index = []
        for i in range(obj_number):
            if data_name[i][k1] != '?':  # 忽略缺失值
                Va_not_mising.append(data_name[i][k1])
            else:
                Va_mising_index.append(i)
        # 计算众数 (出现次数最多的值)
        if Va_not_mising:
            most_common_value = max(set(Va_not_mising), key=Va_not_mising.count)
            for x in Va_mising_index:
                new_data[x][k1] = most_common_value
        else:
            print(f"分类属性 {k1} 没有可用的数据进行填充。")

    # 处理数值属性
    for k2 in range(att_number - real_att - 1, att_number - 1):  # 数值属性的索引
        Va_not_mising = []
        Va_mising_index = []
        for i in range(obj_number):
            if data_name[i][k2] != '?':
                try:
                    # 将非缺失值转换为浮点数
                    Va_not_mising.append(float(data_name[i][k2]))
                except ValueError:
                    print(f"警告: '{data_name[i][k2]}' 不是有效的数值，跳过。")
            else:
                Va_mising_index.append(i)
        # 计算平均值
        if Va_not_mising:
            avg_value = round(sum(Va_not_mising) / len(Va_not_mising), 2)
            for y in Va_mising_index:
                new_data[y][k2] = avg_value
        else:
            print(f"数值属性 {k2} 没有可用的数据进行填充。")
    return new_data

class Dataset:
    def __init__(self, name_ds, number, decision, n_categorical):
        self.name_ds = name_ds
        self.number = number
        self.decision = decision
        self.n_categorical = n_categorical

        self.real_att_number = decision - n_categorical

    def AttributeReduction(self):

        df = pd.read_excel('../datasets/' + self.name_ds + '.xlsx', header=None)
        data_array = np.array(df, dtype=object)
        # 补全属性列的缺失值
        new_data = complete_data(data_array, self.real_att_number)
        missing_ratio = 0.25  # 设定随机缺失的比例，例如 25% 的数据将设置为缺失
        total_rows = len(new_data)  # 获取数据的总行数
        missing_indices = np.random.choice(total_rows, size=int(total_rows * missing_ratio),
                                           replace=False)  # 随机选择需要设置为缺失的行索引
        for idx in missing_indices:
            new_data[idx, -1] = '*'  # 将选定的行设为 '*'，用 '*' 表示缺失
        data = pd.DataFrame(new_data)
        # data = data.iloc[1:].reset_index(drop=True)
        data.to_excel('../Dataset/' + self.name_ds + '(0.25).xlsx', index=False, header=False)

        # print("初始数据：\n", df)
        # print('转换成数组的数据：\n', new_data)
        # print("决策属性随机缺失的数据集：\n", data)












if __name__ == '__main__':
    # print("%s 开始NumericalExperiment")
    StartTime = datetime.datetime.today()
    print("从 {} 开始NumericalExperiment".format(StartTime))
    dic_ds = {
        # 数据集名：[数据集名，数据集样本数，条件属性总数，分类属性总数]
        # '000_example': ['000_example', 10, 6,4],

    }
    for v in dic_ds.values():
        ds = Dataset(*v)
        ds.AttributeReduction()



