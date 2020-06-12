#%%
import os
import pandas as pd

path = os.path.abspath('.')
train_app = pd.read_csv(os.path.join(path, 'train', 'train_app.csv'), sep=',', engine='python')
train_sms = pd.read_csv(os.path.join(path, 'train', 'train_sms.csv'), sep=',', engine='python')
train_user = pd.read_csv(os.path.join(path, 'train', 'train_user.csv'), sep=',', engine='python')
train_voc = pd.read_csv(os.path.join(path, 'train', 'train_voc.csv'), sep=',', engine='python')

#%%

#%%
#description:
# right:4144
# 1: 1962
#%% analyse train_voc
# for time_duration
import seaborn as sns
import matplotlib.pyplot as plt
print(train_voc['call_dur'].describe())
sns.distplot(train_voc['call_dur'])
plt.show()
#%%
# print(train_voc.call_dur.quantile([0.3, 0.7]))
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
new_train_voc = pd.DataFrame(dict_train_voc)
#probablity of Fake
#%%
#get number of suspection
# temp1, temp2, temp3 = [], [], []
temp, temp2 = [], []
e = 10
for name in new_train_voc.phone_no_m:
    x = train_voc.loc[train_voc['phone_no_m'] == name]
    n_1 = x.loc[x.calltype_id == 1]
    n_2 = x.loc[x.calltype_id == 2]
    n_3 = x.loc[x.calltype_id == 3]
    # temp1.append(n_1['label_call_dur'].sum())
    # temp2.append(n_2['label_call_dur'].sum())
    # temp3.append(n_3['label_call_dur'].sum())
    #嫌疑电话中呼入呼出率
    temp.append((e + n_1['label_call_dur'].sum())/(e + n_2['label_call_dur'].sum() + n_3['label_call_dur'].sum()))
    #嫌疑电话数量
    temp2.append(x.label_call_dur.sum())
# print(temp1, temp2, temp3)
print(temp, temp2)
#%%
#merge dataframe
new_train_voc['num_of_sus'] = temp2
new_train_voc['num_of_sus_prob'] = temp

#%% save
new_train = pd.merge(train_user, new_train_voc, how='left', on=['phone_no_m'])
new_train.to_csv('new_train.csv')
