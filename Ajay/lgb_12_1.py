import numpy as np
np.random.seed(444)

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split

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

#lgb_train = lgb.Dataset(X, y_train, categorical_feature=cat_cols)
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'num_threads': 6,
    'num_class': 21,
    'verbose': 1
}

X, X_test, Y, Y_test = train_test_split(X, y_train, test_size=0.25, random_state=666)

lgb_train = lgb.Dataset(X, Y, categorical_feature=cat_cols)
bst_cv = lgb.cv(params=params, train_set=lgb_train, num_boost_round=100, nfold=3, early_stopping_rounds=5, categorical_feature=cat_cols, verbose_eval=True)

print(len(bst_cv['multi_logloss-mean']))
nrounds=len(bst_cv['multi_logloss-mean'])

lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train, categorical_feature=cat_cols)
gbm = lgb.train(params,
                lgb_train,
                valid_sets=lgb_eval,
                num_boost_round=100,
                verbose_eval=True,
                categorical_feature=cat_cols,
                early_stopping_rounds=2)

gbm.best_iteration
pred = gbm.predict(X_test, num_iteration=100)
log_loss(Y_test, pred)
