#%%
import os
import pandas as pd
path = os.path.abspath('.')
# path = '/Users/mayspig/Documents/GitHub/2020SCcompetition'
#%%
new_train = pd.read_csv(os.path.join(path, 'new_train.csv'), sep=',', engine='python')
new_test = pd.read_csv(os.path.join(path, 'new_test.csv'), sep=',', engine='python')
#%%
new_test.isnull().sum()/new_test.shape[0]
#%%

from imblearn.over_sampling import SMOTE
def fillnan(x):
    x['city_name_mean_arup'] = x['city_name_mean_arup'].fillna(new_train.city_name_mean_arup.mean())
    x['county_name_mean_arup'] = x['county_name_mean_arup'].fillna(new_train.county_name_mean_arup.mean())
    cat_col = [i for i in x.columns if i not in ['Unnamed: 0', 'phone_no_m', 'arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911', 'arpu_201912', 'arpu_202001', 'arpu_202002']]
    x = x[cat_col]
    x['isimei'] = x['isimei'].fillna(1)
    x = x.fillna(0)
    return x
def smote_train(train_data):
    cat_col = train_data.columns[train_data.columns !='label']
    # cat_col = [i for i in train_data.columns if i is not 'label']
    data_x = train_data.loc[:, cat_col]
    data_y = train_data.loc[:, 'label']
    sm = SMOTE(random_state=42)
    data_x, data_y = sm.fit_sample(data_x, data_y)
    # n_sample = data_x.shape[0]
    # n_0_sample = data_y.value_counts()[0]
    # n_1_sample = data_y.value_counts()[1]
    # train_data = pd.concat([data_x, data_y], axis=1)
    return data_x, data_y
#%%
train_data = fillnan(new_train)
train, target = smote_train(train_data)
test = fillnan(new_test)

print(train.shape)
print(target.shape)
print(test.shape)

#%%
train.to_csv('train_data.csv')
test.to_csv('test_data.csv')
target.to_csv('target.csv')

#问题
#处理方式不统一，cityname和countyname处理方式不统一