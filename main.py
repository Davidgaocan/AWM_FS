import os
import pandas as pd

# 从不同模块导入testalldata并分别命名以避免冲突
from utiles.testiterTall_npz import testalldata as knn_testalldata
from utiles.testiterTall_npz_alldata_DT import testalldata as dt_testalldata
from utiles.testiterTall_npz_alldata_svm import testalldata as svm_testalldata

if __name__ == '__main__':
    data_dir = './datasets'
    df = pd.read_excel("./xlsx/datasets.xlsx")
    
    for file_name in os.listdir(data_dir):
        data_prefix = os.path.splitext(file_name)[0]
        # 检查文件前缀是否在Excel的Dataset列中且文件为npz格式
        if data_prefix in df['Dataset'].values and file_name.endswith('.npz'):
            file_path = os.path.join(data_dir, file_name)
            # 分别调用三个不同模块的testalldata函数，参数保持一致
            knn_testalldata(file_path, 5, 'datasets', 0.001)
            dt_testalldata(file_path, 5, 'datasets', 0.001)
            svm_testalldata(file_path, 5, 'datasets', 0.001)




