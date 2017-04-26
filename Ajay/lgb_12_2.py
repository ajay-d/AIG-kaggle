import numpy as np
np.random.seed(444)

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss

os.chdir("C:\\Users\\adeonari\\Downloads\\AIG\\data")
os.getcwd()

train = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_12.csv.gz', compression="gzip")

y_train = train['target'].ravel()
X = train.drop(['claim_id', 'loss', 'target'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

cat_cols = [f for f in train.columns if 'cat' in f]

lgb_train = lgb.Dataset(X, y_train, categorical_feature=cat_cols)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    #'boosting_type': 'dart',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_gain_to_split': 0,
    'learning_rate': 0.1,
    #'num_leaves': 127,
    'num_leaves': 32768,
    'bagging_freq': 5,
    'num_threads': 6,
    #'max_bin' : 255,
    'max_bin' : 1024,
    'num_class': 21,
    'verbose': 1
}

bst_cv = lgb.cv(params=params, train_set=lgb_train, num_boost_round=5000, nfold=3, early_stopping_rounds=10, categorical_feature=cat_cols, verbose_eval=True)
#.2476

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    #'boosting_type': 'dart',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_gain_to_split': 0,
    'learning_rate': 0.1,
    'num_leaves': 127,
    'bagging_freq': 5,
    'num_threads': 6,
    'max_bin' : 255,
    'num_class': 21,
    'verbose': 1
}

bst_cv = lgb.cv(params=params, train_set=lgb_train, num_boost_round=5000, nfold=3, early_stopping_rounds=10, categorical_feature=cat_cols, verbose_eval=True)
#.2472

print(len(bst_cv['multi_logloss-mean']))
#nrounds=len(bst_cv['multi_logloss-mean'])
nrounds = round(len(bst_cv['multi_logloss-mean'])/.8)

gbm = lgb.train(params,
                lgb_train,
                num_boost_round=nrounds,
                verbose_eval=True,
                categorical_feature=cat_cols)

pred_final = gbm.predict(X_test, num_iteration=nrounds)

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_final)], axis=1)

df.to_csv('lgblvl1_data12_1.csv.gz', index = False, compression='gzip')