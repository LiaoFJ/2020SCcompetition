#%%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
path = os.path.abspath('.')
#%%
def Reading_train_data(path):
    train_app = pd.read_csv(os.path.join(path, 'train', 'train_app.csv'), sep=',', engine='python')
    train_sms = pd.read_csv(os.path.join(path, 'train', 'train_sms.csv'), sep=',', engine='python')
    train_voc = pd.read_csv(os.path.join(path, 'train', 'train_voc.csv'), sep=',', engine='python')
    train_user = pd.read_csv(os.path.join(path, 'train', 'train_user.csv'), sep=',', engine='python')
    return train_app, train_sms, train_voc, train_user
#%%
def Voc_extraction(train_voc):
    train_voc['label_call_dur'] = train_voc['call_dur'].apply(lambda x: 1 if x < 11 or x > 200 else 0)
    x = train_voc['phone_no_m'].value_counts()
    dict_train_voc = {'phone_no_m': x.index, 'num_of_call': x.values}
    # new dataframe
    # 需不需要normalized
    new_train_voc = pd.DataFrame(dict_train_voc)
    e = 10
    #imei码
    def get_imei_m(x):
        temp = x['imei_m'].value_counts()
        if len(temp) == 1:
            return 0
        return 1
    temp, temp2, temp3 = [], [], []
    #呼入呼出率
    for name in new_train_voc.phone_no_m:
        x = train_voc.loc[train_voc['phone_no_m'] == name]
        n_1 = x.loc[x.calltype_id == 1]
        n_2 = x.loc[x.calltype_id == 2]
        n_3 = x.loc[x.calltype_id == 3]
        temp.append((e + n_1['label_call_dur'].sum())/(e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
    #嫌疑电话数量
        temp2.append(x.label_call_dur.sum())
    #for imei_m
        temp3.append(get_imei_m(x))
    new_train_voc['num_of_sus'] = temp2
    new_train_voc['num_of_sus_prob'] = temp
    new_train_voc['isimei'] = temp3
    col = 'call_dur'
    dict_avg = dict(train_voc.groupby(['phone_no_m']).mean()[col])
    new_train_voc['avg_call_dur'] = new_train_voc['phone_no_m'].map(dict_avg)
    new_train_voc['num_of_call_high'] = new_train_voc['num_of_call'].apply(lambda x: 1 if x >= 50 else 0)
    return new_train_voc
#%%
def App_extraciton(train_app):
    # analyse train_app
    # app_counts = train_app['month_id'].value_counts()
    x = train_app['phone_no_m'].value_counts()
    dict_train_app = {'phone_no_m': x.index}

    # new dataframe
    new_train_app = pd.DataFrame(dict_train_app)
    temp_app, temp_app_name = [], []
    # 后期可以把每个月的特征都拆接下来
    for name in new_train_app.phone_no_m:
        x = train_app.loc[train_app['phone_no_m'] == name]
        temp_app.append(x['flow'].sum())
    new_train_app['flow'] = temp_app
    return new_train_app

def User_extraction(train_user, col):
    # 构建消费统计特征
    dict_city = dict(train_user.groupby(['city_name']).mean()[col])
    dict_county = dict(train_user.groupby(['county_name']).mean()[col])
    train_user['city_name_mean_arup'] = train_user['city_name'].map(dict_city)
    train_user['county_name_mean_arup'] = train_user['county_name'].map(dict_county)
    train_user[col] = train_user[col].fillna(0)
    # 判断当月消费记录是否为空
    train_user['arup_null'] = train_user[col].apply(lambda x: 1 if x == 0 else 0)
    # 是否属于高消费人群
    train_user['arup_high'] = train_user[col].apply(lambda x: 1 if x >= 150 else 0)
    # 转为one-hot编码
    cat_col = ['city_name', 'county_name']
    for i in tqdm(cat_col):
        lbl = LabelEncoder()
        train_user[i] = lbl.fit_transform(train_user[i].astype(str))

    return train_user

#%%get train
col = 'arpu_202003'

print('reading data')
train_app, train_sms, train_voc, train_user = Reading_train_data(path)
print('app extraction')
new_train_app = App_extraciton(train_app)
print('voc extraction')
new_train_voc = Voc_extraction(train_voc)
print('user extraction')
new_train_user = User_extraction(train_user, col)
print('merge data')
new_train = pd.merge(new_train_user, new_train_voc, how='left', on=['phone_no_m'])
new_train = pd.merge(new_train, new_train_app, how='left', on=['phone_no_m'])

#%%save
print('save data')
new_train.to_csv('new_train.csv')

