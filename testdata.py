#%%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
path = os.path.abspath('.')

#%%
def Reading_test_data(path):
    test_app = pd.read_csv(os.path.join(path, 'test', 'test_app.csv'), sep=',', engine='python', quotechar='"')
    test_sms = pd.read_csv(os.path.join(path, 'test', 'test_sms.csv'), sep=',', engine='python', quotechar='"')
    test_voc = pd.read_csv(os.path.join(path, 'test', 'test_voc.csv'), sep=',', engine='python', quotechar='"')
    test_user = pd.read_csv(os.path.join(path, 'test', 'test_user.csv'), sep=',', engine='python', quotechar='"')
    return test_app, test_sms, test_voc, test_user
#%%
def Voc_extraction(test_voc):
    test_voc['label_call_dur'] = test_voc['call_dur'].apply(lambda x: 1 if x < 11 or x > 200 else 0)
    x = test_voc['phone_no_m'].value_counts()
    dict_test_voc = {'phone_no_m': x.index, 'num_of_call': x.values}
    # new dataframe
    # 需不需要normalized
    new_test_voc = pd.DataFrame(dict_test_voc)
    e = 10
    #imei码
    def get_imei_m(x):
        temp = x['imei_m'].value_counts()
        if len(temp) == 1:
            return 0
        return 1
    temp, temp2, temp3 = [], [], []
    #呼入呼出率
    for name in new_test_voc.phone_no_m:
        x = test_voc.loc[test_voc['phone_no_m'] == name]
        n_1 = x.loc[x.calltype_id == 1]
        n_2 = x.loc[x.calltype_id == 2]
        n_3 = x.loc[x.calltype_id == 3]
        temp.append((e + n_1['label_call_dur'].sum())/(e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
    #嫌疑电话数量
        temp2.append(x.label_call_dur.sum())
    #for imei_m
        temp3.append(get_imei_m(x))
    new_test_voc['num_of_sus'] = temp2
    new_test_voc['num_of_sus_prob'] = temp
    new_test_voc['isimei'] = temp3
    col = 'call_dur'
    dict_avg = dict(test_voc.groupby(['phone_no_m']).mean()[col])
    new_test_voc['avg_call_dur'] = new_test_voc['phone_no_m'].map(dict_avg)
    new_test_voc['num_of_call_high'] = new_test_voc['num_of_call'].apply(lambda x: 1 if x >= 50 else 0)
    return new_test_voc
#%%
def App_extraciton(test_app):
    # analyse test_app
    # app_counts = test_app['month_id'].value_counts()
    x = test_app['phone_no_m'].value_counts()
    dict_test_app = {'phone_no_m': x.index}

    # new dataframe
    new_test_app = pd.DataFrame(dict_test_app)
    temp_app, temp_app_name = [], []
    # 后期可以把每个月的特征都拆接下来
    for name in new_test_app.phone_no_m:
        x = test_app.loc[test_app['phone_no_m'] == name]
        temp_app.append(x['flow'].sum())
    new_test_app['flow'] = temp_app
    return new_test_app

def User_extraction(test_user, col):
    # 构建消费统计特征
    dict_city = dict(test_user.groupby(['city_name']).mean()[col])
    dict_county = dict(test_user.groupby(['county_name']).mean()[col])
    test_user['city_name_mean_arup'] = test_user['city_name'].map(dict_city)
    test_user['county_name_mean_arup'] = test_user['county_name'].map(dict_county)
    test_user[col] = test_user[col].fillna(0)
    # 判断当月消费记录是否为空
    test_user['arup_null'] = test_user[col].apply(lambda x: 1 if x == 0 else 0)
    # 是否属于高消费人群
    test_user['arup_high'] = test_user[col].apply(lambda x: 1 if x >= 150 else 0)
    # 转为one-hot编码
    cat_col = ['city_name', 'county_name']
    for i in tqdm(cat_col):
        lbl = LabelEncoder()
        test_user[i] = lbl.fit_transform(test_user[i].astype(str))

    return test_user

#%%get test
col = 'arpu_202004'
print('reading data')
test_app, test_sms, test_voc, test_user = Reading_test_data(path)
print('app extraction')
new_test_app = App_extraciton(test_app)
print('voc extraction')
new_test_voc = Voc_extraction(test_voc)
print('user extraction')
new_test_user = User_extraction(test_user, col)
print('merge data')
new_test = pd.merge(new_test_user, new_test_voc, how='left', on=['phone_no_m'])
new_test = pd.merge(new_test, new_test_app, how='left', on=['phone_no_m'])

#%%save
print('save data')
new_test.to_csv('new_test.csv')

