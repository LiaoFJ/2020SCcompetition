#%%
import os
import pandas as pd
path = os.path.abspath('.')

new_train = pd.read_csv(os.path.join(path, 'new_train.csv'), sep=',', engine='python')
new_test = pd.read_csv(os.path.join(path, 'new_test.csv'), sep=',', engine='python')
#%%
new_test.isnull().sum()/new_test.shape[0]
#%%
def fillnan(x):
    x['city_name_mean_arup'] = x['city_name_mean_arup'].fillna(new_train.city_name_mean_arup.mean())
    x['county_name_mean_arup'] = x['county_name_mean_arup'].fillna(new_train.county_name_mean_arup.mean())
    cat_col = [i for i in x.columns if i not in ['Unnamed: 0', 'label','phone_no_m','arpu_201908','arpu_201909','arpu_201910','arpu_201911','arpu_201912','arpu_202001','arpu_202002']]
    x = x[cat_col]
    x['isimei'] = x['isimei'].fillna(1)
    x = x.fillna(0)
    return x
#%%
train_data = fillnan(new_train)
test_data = fillnan(new_test)
target = new_train['label']
#%%
train_data.to_csv('train_data.csv')
test_data.to_csv('test_data.csv')
target.to_csv('target.csv')