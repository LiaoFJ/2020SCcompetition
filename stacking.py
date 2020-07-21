#%%

import os

path = os.path.abspath('.')

from catboost import CatBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
train = pd.read_csv(os.path.join(path, 'data', 'train_data.csv'), sep=',', engine='python', index_col=0)
target = pd.read_csv(os.path.join(path, 'data', 'target.csv'), sep=',', engine='python', index_col=0, header = None)
test = pd.read_csv(os.path.join(path, 'data', 'test_data.csv'), sep=',', engine='python', index_col=0)
test = test.rename(columns={"arpu_202004":"arpu_202003"})
target = target.fillna(0)


x_train, x_validation, y_train, y_validation = train_test_split(train, target, test_size = 0.3, random_state=12)
#%%
from mlxtend.classifier import StackingClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import lightgbm as lgb
#%%
print('model loading')
baseline1 = XGBClassifier(  learning_rate =0.1,
                            n_estimators=500,
                            max_depth=6,
                            min_child_weight=1,
                            gamma=0,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective= 'binary:logistic',
                            nthread=4,
                            scale_pos_weight=1,
                            seed=27)
baseline2 = lgb.LGBMClassifier(random_state=626,
                               n_estimators=500,
                               learning_rate=0.033,
                               boosting_type='gbdt',
                               max_depth=-1,
                               num_leaves=70,
                               colsample_bytree=0.8,
                               subsample = 0.8,
                               lambda_l1 = 0.1
                             #  labda_l2 = 0.2,
                               )
baseline3 = RandomForestClassifier(n_estimators = 500, oob_score =True ,n_jobs = 1,random_state =1)
baseline4 = CatBoostClassifier(iterations=500, depth=6,learning_rate=0.033, loss_function='Logloss',
                            logging_level='Verbose')
baseline5 = AdaBoostClassifier()
baseline6 = GaussianNB()
baseline7 = SVC(kernel = 'rbf', class_weight = 'balanced')
lr = XGBClassifier()

stackmodel = StackingClassifier(classifiers=[baseline1, baseline2, baseline3, baseline4, baseline5, baseline6, baseline7],

                        meta_classifier=lr)

#%%
for basemodel,label in zip([baseline1, baseline2, baseline3, baseline4, baseline5, baseline6, baseline7, stackmodel],
                           ['xgboost',
                            'lightgbm',
                            'Random Forest',
                            'Catboost',
                            'AdaBoost',
                            'GaussianNB',
                            'SVC',
                            'stack']):

    scores = model_selection.cross_val_score(basemodel, train, target, cv=5, scoring='accuracy')
#%%
stackmodel.fit(train, target)
predict = stackmodel.predict(test)
#%%
print('data saving')
predict = pd.DataFrame(predict)
predict.to_csv('./data/stacking.csv', encoding='utf-8', index=None)


