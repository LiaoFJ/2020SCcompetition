#%%
import os
import pandas as pd

path = os.path.abspath('.')
train_app = pd.read_csv(os.path.join(path, 'train', 'train_app.csv'), sep=',', engine='python', nrows=10000)
train_sms = pd.read_csv(os.path.join(path, 'train', 'train_sms.csv'), sep=',', engine='python', nrows=10000)
train_user = pd.read_csv(os.path.join(path, 'train', 'train_user.csv'), sep=',', engine='python', nrows=1000)
train_voc = pd.read_csv(os.path.join(path, 'train', 'train_voc.csv'), sep=',', engine='python', nrows=10000)

#%%
from tqdm import tqdm
cat_col = ['city_name', 'county_name']
for i in tqdm(cat_col):
    x = train_user[i].drop_duplicates().values.tolist()
    b = dict()
    for i, item in enumerate(x):
        b[i] = item
    print(b)

#%%
#description:
# right:4144
# 1: 1962
#%% analyse train_voc
# for time_duration
# import seaborn as sns
# import matplotlib.pyplot as plt
# print(train_voc['call_dur'].describe())
# sns.distplot(train_voc['call_dur'])
# plt.show()

#%%

print(train_voc.call_dur.quantile([0.1, 0.9]))
#%%
#0.15    14.0
#0.30    22.0
#0.75    79.0
#异常值
train_voc['label_call_dur']=train_voc['call_dur'].apply(lambda x: 1 if x < 22 or x > 67 else 0)
#%%
x = train_voc['phone_no_m'].value_counts()
dict_train_voc = {'phone_no_m': x.index, 'num_of_call': x.values}

#new dataframe
#需不需要normalized
new_train_voc = pd.DataFrame(dict_train_voc)
#%%
#probablity of Fake
#get number of suspection
# temp1, temp2, temp3 = [], [], []

def get_imei_m(x):
    temp = x['imei_m'].value_counts()
    if len(temp) == 1:
        return 0
    return 1
temp, temp2, temp3, temp_num= [], [], [], []
calling_month = []
temp_re = []
e = 10
for name in new_train_voc.phone_no_m:
    x = train_voc.loc[train_voc['phone_no_m'] == name]
    n_1 = x.loc[x.calltype_id == 1]
    n_2 = x.loc[x.calltype_id == 2]
    n_3 = x.loc[x.calltype_id == 3]
    temp_num.append(x['opposite_no_m'].nunique())
    # temp1.append(n_1['label_call_dur'].sum())
    # temp2.append(n_2['label_call_dur'].sum())
    # temp3.append(n_3['label_call_dur'].sum())
    #嫌疑电话中呼入呼出率
    temp.append((e + n_1['label_call_dur'].sum())/(e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
    #正常的呼入呼出比
    temp_re.append((e + n_1.shape[0])/(e + n_2.shape[0] + n_3.shape[0]))
    #嫌疑电话数量
    temp2.append(x.label_call_dur.sum() / x['start_datetime'].apply(lambda x: x[:len('2020-03')]).nunique())
    #for imei_m
    temp3.append(get_imei_m(x))

    #通话的月数：
    calling_month.append(x['start_datetime'].apply(lambda x: x[:len('2020-03')]).nunique())
new_train_voc['calling_month'] = calling_month
new_train_voc['num_of_call'] = new_train_voc.apply(lambda x : x['num_of_call'] / x['calling_month'], axis=1)



col = 'call_dur'
dict_avg = dict(train_voc.groupby(['phone_no_m']).mean()[col])
new_train_voc['avg_call_dur'] = new_train_voc['phone_no_m'].map(dict_avg)
# print(temp1, temp2, temp3)
#%%
print(temp_re)
#%%
#merge dataframe
new_train_voc['num_of_sus'] = temp2
new_train_voc['num_of_sus_prob'] = temp
new_train_voc['isimei'] = temp3
#%%
print(new_train_voc.num_of_sus.quantile([0.1, 0.9]))
#%%
col = ['arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911', 'arpu_201912', 'arpu_202001', 'arpu_202002', 'arpu_202003']
train_user['arpu_202003'] = train_user[col].mean(1)
train_user['isnan'] = train_user[col].isnull().any(axis=1)






#%%
#analyse train_app
app_counts = train_app['month_id'].value_counts()

#%%
import numpy as np
x = train_app['phone_no_m'].value_counts()
dict_train_app = {'phone_no_m': x.index}

#new dataframe
new_train_app = pd.DataFrame(dict_train_app)
temp_app, temp_app_name, temp_num = [], [], []
#后期可以把每个月的特征都拆接下来
for name in new_train_app.phone_no_m:
    x = train_app.loc[train_app['phone_no_m'] == name]
    temp_app.append(x['flow'].sum())
    temp_num.append(x['month_id'].nunique())
    x = np.array(temp_app) / np.array(temp_num)

    # total_sms_month = x['month_id'].apply(lambda x: x[:len('2019/12')]).unique()
new_train_app['flow'] = temp_app
new_train_app['num_month'] = temp_num
new_train_app['qweqweqwe'] = x.tolist()

num_of_app = dict(train_app.groupby(['phone_no_m']).nunique()['busi_name'])
new_train_app['num_of_app'] = new_train_app['phone_no_m'].map(num_of_app)
#%%


#%%save
new_train = pd.merge(train_user, new_train_voc, how='left', on=['phone_no_m'])
new_train = pd.merge(new_train, new_train_app, how='left', on=['phone_no_m'])




#%%
#train_user
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
col='arpu_202003'
#构建消费统计特征
dict_city = dict(new_train.groupby(['city_name']).mean()[col])
dict_county = dict(new_train.groupby(['county_name']).mean()[col])
new_train['city_name_mean_arup'] = new_train['city_name'].map(dict_city)
new_train['county_name_mean_arup'] = new_train['county_name'].map(dict_county)
new_train[col]=new_train[col].fillna(0)
#判断当月消费记录是否为空
new_train['arup_null']=new_train[col].apply(lambda x:1 if x ==0  else 0)
#是否属于高消费人群
new_train['arup_high']=new_train[col].apply(lambda x:1 if x  >=500  else 0)
#转为one-hot编码
cat_col = ['city_name', 'county_name']
for i in tqdm(cat_col):
    lbl = LabelEncoder()
    new_train[i] = lbl.fit_transform(new_train[i].astype(str))

new_train.to_csv('new_train.csv')

#%%
dict_city = dict()
for key, value in zip(new_train['city_name'], new_train['city_name_mean_arup']):
    dict_city[key] = value
print(dict_city)
#这个是生成的字典

#%%
#可视化
import seaborn as sns
import matplotlib.pyplot as plt
print(new_train['arpu_202003'].describe())
sns.distplot(new_train['arpu_202003'])
plt.show()
print(new_train.arpu_202003.quantile([0.995]))

