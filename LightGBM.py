#%%
import pandas as pd
import os
import numpy as np
path = os.path.abspath('.')
import lightgbm as lgb
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import log_loss
#%%
train = pd.read_csv(os.path.join(path, 'data', 'train_data.csv'), sep=',', engine='python', index_col=0)
target = pd.read_csv(os.path.join(path, 'data', 'target.csv'), sep=',', engine='python', index_col=0, header = None)
test = pd.read_csv(os.path.join(path, 'data', 'test_data.csv'), sep=',', engine='python', index_col=0)

#%%

target_value =target.values.flatten()


#%%

params = {'num_leaves': 85, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 22,
          'max_depth': -1,
          'objective': 'binary', #定义的目标函数
          'learning_rate': 0.03,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.05,             #l1正则
          # 'lambda_l2': 0.001,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'binary_logloss', 'auc'},  ##评价函数选择
          "random_state": 2019
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

folds = KFold(n_splits=5, shuffle=True, random_state=2019)

prob_oof = np.zeros((train.shape[0], ))

feature_importance_df = pd.DataFrame()
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train)):
    print("fold {}".format(fold_ + 1))
    trn_data = lgb.Dataset(train.iloc[trn_idx], label=target_value[trn_idx])
    val_data = lgb.Dataset(train.iloc[val_idx], label=target_value[val_idx])

clf = lgb.train(params,
                trn_data,
                valid_sets=[trn_data, val_data],
                verbose_eval=20,
                num_boost_round=50000,
                early_stopping_rounds=600)
prob_oof[val_idx] = clf.predict(train.iloc[val_idx], num_iteration=clf.best_iteration)


#%%
result = clf.predict(test, num_iteration=clf.best_iteration)
#%%
result[result<0.5] = 0
result[result>=0.5] = 1
#%%
sub = pd.read_csv(os.path.join(path, 'data', 'submit_example.csv'), sep=',', engine='python')
sub.iloc[:,1] = result
#%%
sub.to_csv('./data/lgb.csv', encoding='utf-8', index=None)