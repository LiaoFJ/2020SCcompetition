# 2020SCcompetition

此文件为2020年首届四川电话诈骗初赛 本小组的源码
分享出来仅供参考与使用

# 下载数据集
使用的方法是运行 requist_try.py 
python download_gdrive.py GoogleFileID /path/xxxx.file
下载数据集合至当前文件夹

# 文件夹的位置关系如下：

--当前文件夹

 |-DataExtraction.py
 |-datasetupdate.py
 |-LightGBM.py
 |-stacking.py
 |--train
    |-train_app.csv
    |-train_sms.csv
    |-train_user.csv
    |-train_voc.csv
   
 |--test
    test_app.csv
    test_sms.csv
    test_user.csv
    test_voc.csv
    
 |--data
    submit_example.csv
# 依次运行：

## DataExtraction.py
获取（得到）
new_train.csv
new_test.csv
文件

## datasetupdate.py
获取（得到）
train_data.csv
test_data.csv
target.csv

# 最后的学习

文档中提供了两种学习的方式
第一种为基础的lightgbm的方式
获得的准确率大概在0.88左右
第二种是将多个基础的学习器stacking
获得的准确率在验证集合中大约0.91
