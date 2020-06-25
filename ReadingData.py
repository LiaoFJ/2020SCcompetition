# %%
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

path = os.path.abspath('.')
# path = '/Users/mayspig/Desktop/竞赛/诈骗电话竞赛资料/诈骗电话号码识别-0527'


# %%
def Reading_train_data(path):
    # train_app = pd.read_csv(os.path.join(path, 'train', 'train_app.csv'), sep=',', engine='python', nrows=10000)
    # train_sms = pd.read_csv(os.path.join(path, 'train', 'train_sms.csv'), sep=',', engine='python', nrows=10000)
    # train_voc = pd.read_csv(os.path.join(path, 'train', 'train_voc.csv'), sep=',', engine='python', nrows=10000)
    # train_user = pd.read_csv(os.path.join(path, 'train', 'train_user.csv'), sep=',', engine='python', nrows=1000)
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
    e = 5
    # imei码
    def get_imei_m(x):
        temp = x['imei_m'].value_counts()
        if len(temp) == 1:
            return 0
        return 1
    temp, temp2, temp3, temp_num_pr = [], [], [], []
    # 呼入呼出率
    temp_re = []
    for name in new_train_voc.phone_no_m:
        x = train_voc.loc[train_voc['phone_no_m'] == name]
        n_1 = x.loc[x.calltype_id == 1]
        n_2 = x.loc[x.calltype_id == 2]
        n_3 = x.loc[x.calltype_id == 3]
        temp.append((n_1['label_call_dur'].sum() + e ) / (e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
        temp_num_pr.append(x['opposite_no_m'].nunique())
        #正常呼入呼出比
        temp_re.append((e + n_1.shape[0]) / (e + n_2.shape[0] + n_3.shape[0]))
        # 嫌疑电话数量
        temp2.append(x.label_call_dur.sum())
        # for imei_m
        temp3.append(get_imei_m(x))
    new_train_voc['num_of_sus'] = temp2
    new_train_voc['num_of_sus_prob'] = temp
    new_train_voc['isimei'] = temp3
    new_train_voc['num_of_pr'] = temp_num_pr
    new_train_voc['num_of_prob'] = temp_re
    col = 'call_dur'
    dict_avg = dict(train_voc.groupby(['phone_no_m']).mean()[col])
    new_train_voc['avg_call_dur'] = new_train_voc['phone_no_m'].map(dict_avg)
    new_train_voc['num_of_call_sus_high'] = new_train_voc['num_of_sus'].apply(lambda x: 1 if x >= 357 else 0)
    return new_train_voc
    #还可以细化，平均每日电话数量

# %%
def App_extraciton(train_app):
    # analyse train_app
    # app_counts = train_app['month_id'].value_counts()
    x = train_app['phone_no_m'].value_counts()
    dict_train_app = {'phone_no_m': x.index}

    # new dataframe
    new_train_app = pd.DataFrame(dict_train_app)
    temp_app, temp_app_name, temp_num = [], [], []
    # 后期可以把每个月的特征都拆接下来
    for name in new_train_app.phone_no_m:
        x = train_app.loc[train_app['phone_no_m'] == name]
        temp_app.append(x['flow'].sum())
        temp_num.append(x['month_id'].nunique())
    new_train_app['flow'] = temp_app
    #平均每月
    new_train_app['num_month'] = temp_num
    #app数量
    num_of_app = dict(train_app.groupby(['phone_no_m']).nunique()['busi_name'])
    new_train_app['num_of_app'] = new_train_app['phone_no_m'].map(num_of_app)

    return new_train_app

def User_extraction(train_user, col):
    # 构建消费统计特征
    colsum = ['arpu_201908', 'arpu_201909', 'arpu_201910', 'arpu_201911', 'arpu_201912', 'arpu_202001', 'arpu_202002',
           'arpu_202003']
    train_user[col] = train_user[colsum].mean(1)
    dict_city = dict(train_user.groupby(['city_name']).mean()[col])
    dict_county = dict(train_user.groupby(['county_name']).mean()[col])
    train_user['city_name_mean_arup'] = train_user['city_name'].map(dict_city)
    train_user['county_name_mean_arup'] = train_user['county_name'].map(dict_county)
    # 判断是否有月消费记录是否为空
    train_user['arup_null'] = train_user[colsum].isnull().any(axis=1)
    #这个不确定到底有没有用
    # train_user[col] = train_user[col].fillna(0)

    # 是否属于高消费人群
    train_user['arup_high'] = train_user[col].apply(lambda x: 1 if x >= 150 else 0)
    # 转为one-hot编码
    cat_col = ['city_name', 'county_name']
    for i in tqdm(cat_col):
        lbl = LabelEncoder()
        train_user[i] = lbl.fit_transform(train_user[i].astype(str))
    #计算月均消费以及有没有nan

    return train_user


def Sms_extraciton(train_sms):
    train_sms2 = train_sms.groupby('phone_no_m')

    train_sms3 = pd.DataFrame(
        columns=['phone_no_m', 'total_receive', 'total_send', 'ratio(send/receive)', 'total_sms_month', 'total_sms_day',
                 'month_average_send', 'day_average_send', 'month_average_receive', 'day_average_receive'])
    i = 0
    for phone_no_m, value in train_sms2:
        type1 = value[value['calltype_id'] == 1]
        type2 = value[value['calltype_id'] == 2]
        # 两种短信方式的次数统计
        total_receive = len(type1)
        total_send = len(type2)
        # 接收/发送 率
        if (total_receive != 0) & (total_send != 0):
            ratio = total_receive / total_send
        else:
            ratio = -1

        # 有效短信总月数
        total_sms_month = len(value['request_datetime'].apply(lambda x: x[:len('2020-03')]).unique())
        # 有效短信总天数
        total_sms_day = len(value['request_datetime'].apply(lambda x: x.split(' ')[0]).unique())

        new = pd.DataFrame({'phone_no_m': phone_no_m,
                            'total_receive': total_receive, # 接收短信数
                            'total_send': total_send, # 发送短信数
                            'ratio(send/receive)': ratio, #（接收/发送）率
                            'total_sms_month': total_sms_month, # 有效短信总月数
                            'total_sms_day': total_sms_day,# 有效短信总天数
                            'month_average_send': total_send / total_sms_month, # 平均月发送量
                            'day_average_send': total_send / total_sms_day,# 平均天发送量
                            'month_average_receive': total_receive / total_sms_month, # 平均月接收量
                            'day_average_receive': total_receive / total_sms_day},# 平均日接收量
                           index=[i])
        train_sms3 = train_sms3.append(new, ignore_index=True)
    return train_sms3


# %%get train
col = 'arpu_202003'
print('reading data')
train_app, train_sms, train_voc, train_user = Reading_train_data(path)
print('app extraction')
new_train_app = App_extraciton(train_app)
print('voc_extraction')
new_train_voc = Voc_extraction(train_voc)
print('User_extracition')
new_train_user = User_extraction(train_user, col)
print('Sms_extracition')
new_train_sms = Sms_extraciton(train_sms)
print('merge')
new_train = pd.merge(new_train_user, new_train_voc, how='left', on=['phone_no_m'])
new_train = pd.merge(new_train, new_train_app, how='left', on=['phone_no_m'])
new_train = pd.merge(new_train, new_train_sms, how='left', on=['phone_no_m'])

# %%save
print('save')
new_train.to_csv('new_train.csv')
