
import numpy as np
np.random.seed(666)

import os
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

os.chdir("C:\\Users\\adeonari\\Downloads\\numerai_datasets")
os.getcwd()

train = pd.read_csv("numerai_training_data.csv")
test = pd.read_csv("numerai_tournament_data.csv")

y_train = train['target'].ravel()
X = train.drop(['target'], axis=1)
X_test = test.drop(['t_id'], axis=1)

X.shape
X_test.shape

test_id = test['t_id'].values

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_gain_to_split': 0,
    'learning_rate': 0.03,
    'num_leaves': 127,
    #'num_leaves': 1023,
    'bagging_freq': 5,
    'num_threads': 6,
    'max_bin' : 255,
    #'max_bin' : 511,
    #'num_class': 21,
    'verbose': 1
}

nfolds = 4
kf = KFold(nfolds, shuffle=True)

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

nrounds = 1000
pred_final = np.zeros(X_test.shape[0])
j = 1

for train_index, test_index in kf.split(X):
    X_model, X_oos = X.loc[train_index], X.loc[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    pred_oos = np.zeros(shape=(y_oos.shape[0], 1))
    lgb_train = lgb.Dataset(X_model, y_model)
    lgb_eval = lgb.Dataset(X_oos, y_oos, reference=lgb_train)
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=nrounds,
                    valid_sets=lgb_eval,
                    verbose_eval=True,
                    early_stopping_rounds=10)
    pred_final += gbm.predict(X_test, num_iteration=gbm.best_iteration)
    pred_oos = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
    df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
    j += 1

df_results
np.mean(df_results['logloss'])
#.6790 @127 / 500 rounds
#.66809 @1023 / 1000 rounds

pred_final /= nfolds

df = pd.DataFrame(test_id, columns = ['t_id'])
df = pd.concat([df, pd.DataFrame(pred_final)], axis=1)

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'binary_logloss',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_gain_to_split': 0,
    'learning_rate': 0.01,
    #'num_leaves': 127,
    'num_leaves': 1023,
    'bagging_freq': 5,
    'num_threads': 6,
    'max_bin' : 255,
    #'max_bin' : 511,
    #'num_class': 21,
    'verbose': 1
}

nfolds = 4
kf = KFold(nfolds, shuffle=True)

df_results['logloss'] = np.zeros(df_results.shape[0])

nrounds = 1000
pred_final = np.zeros(X_test.shape[0])
j = 1

for train_index, test_index in kf.split(X):
    X_model, X_oos = X.loc[train_index], X.loc[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    pred_oos = np.zeros(shape=(y_oos.shape[0], 1))
    lgb_train = lgb.Dataset(X_model, y_model)
    lgb_eval = lgb.Dataset(X_oos, y_oos, reference=lgb_train)
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=nrounds,
                    valid_sets=lgb_eval,
                    verbose_eval=True,
                    early_stopping_rounds=10)
    pred_final += gbm.predict(X_test, num_iteration=gbm.best_iteration)
    pred_oos = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
    df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
    j += 1

df_results
np.mean(df_results['logloss'])
#.6790 @127 / 500 rounds
#.66809 @1023 / 1000 rounds

pred_final /= nfolds

df = pd.concat([df, pd.DataFrame(pred_final)], axis=1)

df.to_csv('lgbm_1.csv.gz', index = False, compression='gzip')