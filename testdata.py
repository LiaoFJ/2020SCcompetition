#%%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
path = os.path.abspath('.')
#path = '/Users/mayspig/Desktop/竞赛/诈骗电话竞赛资料/诈骗电话号码识别-0527'
#%%
def Reading_test_data(path):
    test_app = pd.read_csv(os.path.join(path, 'test', 'test_app.csv'), sep=',', engine='python')
    test_sms = pd.read_csv(os.path.join(path, 'test', 'test_sms.csv'), sep=',', engine='python')
    test_voc = pd.read_csv(os.path.join(path, 'test', 'test_voc.csv'), sep=',', engine='python')
    test_user = pd.read_csv(os.path.join(path, 'test', 'test_user.csv'), sep=',', engine='python')
    return test_app, test_sms, test_voc, test_user
#%%
def Voc_extraction(test_voc):
    test_voc['label_call_dur'] = test_voc['call_dur'].apply(lambda x: 1 if x < 11 or x > 200 else 0)
    x = test_voc['phone_no_m'].value_counts()
    dict_test_voc = {'phone_no_m': x.index, 'num_of_call': x.values}
    # new dataframe
    # 需不需要normalized
    new_test_voc = pd.DataFrame(dict_test_voc)
    e = 5
    #imei码
    def get_imei_m(x):
        temp = x['imei_m'].value_counts()
        if len(temp) == 1:
            return 0
        return 1
    temp, temp2, temp3, temp_num_pr = [], [], [], []
    #呼入呼出率
    temp_re = []
    for name in new_test_voc.phone_no_m:
        x = test_voc.loc[test_voc['phone_no_m'] == name]
        n_1 = x.loc[x.calltype_id == 1]
        n_2 = x.loc[x.calltype_id == 2]
        n_3 = x.loc[x.calltype_id == 3]
        temp.append((n_1['label_call_dur'].sum() + e ) / (e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
        temp_num_pr.append(x['opposite_no_m'].nunique())

    #嫌疑电话数量
        temp2.append(x.label_call_dur.sum())
        #正常呼入呼出比
        temp_re.append((e + n_1.shape[0]) / (e + n_2.shape[0] + n_3.shape[0]))
    #for imei_m
        temp3.append(get_imei_m(x))
    new_test_voc['num_of_sus'] = temp2
    new_test_voc['num_of_sus_prob'] = temp
    new_test_voc['isimei'] = temp3
    test_user['num_of_pr'] = temp_num_pr
    new_test_voc['num_of_prob'] = temp_re
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
    temp_app, temp_app_name, temp_num = [], [], []
    # 后期可以把每个月的特征都拆接下来
    for name in new_test_app.phone_no_m:
        x = test_app.loc[test_app['phone_no_m'] == name]
        temp_app.append(x['flow'].sum())
        temp_num.append(x['month_id'].nunique())
    new_test_app['flow'] = temp_app
    #平均每月
    new_test_app['num_month'] = temp_num
    #app数量
    num_of_app = dict(test_app.groupby(['phone_no_m']).nunique()['busi_name'])
    new_test_app['num_of_app'] = new_test_app['phone_no_m'].map(num_of_app)
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


def Sms_extraciton(test_sms):
    test_sms2 = test_sms.groupby('phone_no_m')

    test_sms3 = pd.DataFrame(
        columns=['phone_no_m', 'total_receive', 'total_send', 'ratio(send/receive)', 'total_sms_month', 'total_sms_day',
                 'month_average_send', 'day_average_send', 'month_average_receive', 'day_average_receive'])
    i = 0
    for phone_no_m, value in test_sms2:
        type1 = value[value['calltype_id'] == 1]
        type2 = value[value['calltype_id'] == 2]
        # 两种短信方式的次数统计
        total_receive = len(type1)
        total_send = len(type2)
        # 接收/发送
        if (total_receive != 0) & (total_send != 0):
            ratio = total_receive / total_send
        else:
            ratio = -1

        # 有效发送短信总月数
        total_sms_month = len(value['request_datetime'].apply(lambda x: x[:len('2020-03')]).unique())
        # 有效发送短信总天数
        total_sms_day = len(value['request_datetime'].apply(lambda x: x.split(' ')[0]).unique())

        new = pd.DataFrame({'phone_no_m': phone_no_m,
                            'total_receive': total_receive,
                            'total_send': total_send,
                            'ratio(send/receive)': ratio,
                            'total_sms_month': total_sms_month,
                            'total_sms_day': total_sms_day,
                            'month_average_send': total_send / total_sms_month,
                            'day_average_send': total_send / total_sms_day,
                            'month_average_receive': total_receive / total_sms_month,
                            'day_average_receive': total_receive / total_sms_day},
                           index=[i])
        test_sms3 = test_sms3.append(new, ignore_index=True)
    return test_sms3


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
print('sms extraction')
new_test_sms = Sms_extraciton(test_sms)

print('merge data')
new_test = pd.merge(new_test_user, new_test_voc, how='left', on=['phone_no_m'])
new_test = pd.merge(new_test, new_test_app, how='left', on=['phone_no_m'])
new_test = pd.merge(new_test, new_test_sms, how='left', on=['phone_no_m'])

#%%save
print('save data')
new_test.to_csv('new_test.csv')
