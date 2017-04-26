import numpy as np
np.random.seed(666)

import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

os.chdir("C:\\Users\\adeonari\\Downloads\\AIG\\data")
os.getcwd()

train = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_12.csv.gz', compression="gzip")

cat_cols = [f for f in train.columns if 'cat' in f]
#cat_cols.insert(0,"claim_id")

y_train = train['target'].ravel()
X = train[cat_cols]
X_test = test[cat_cols]

train_id = train['claim_id'].values
test_id = test['claim_id'].values

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
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
    'num_class': 21,
    'verbose': 1
}

#To store stacked values
#train_stack = pd.DataFrame(train_id, columns = ['claim_id'])
#test_stack = pd.DataFrame(test_id, columns = ['claim_id'])
#train_stack = pd.concat([train_stack, pd.DataFrame(np.zeros(shape=(train_stack.shape[0], 21)))], axis=1)
#test_stack = pd.concat([test_stack, pd.DataFrame(np.zeros(shape=(test_stack.shape[0], 21)))], axis=1)
train_stack = pd.DataFrame(np.zeros(shape=(train.shape[0], 21)))
test_stack = pd.DataFrame(np.zeros(shape=(test.shape[0], 21)))

nfolds = 4
kf = KFold(nfolds, shuffle=True)

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

nrounds = 500
pred_final = np.zeros(shape=(X_test.shape[0], 21))
j = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.loc[train_index], X.loc[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    pred_oos = np.zeros(shape=(y_oos.shape[0], 21))
    lgb_train = lgb.Dataset(X_model, y_model, categorical_feature=cat_cols)
    lgb_eval = lgb.Dataset(X_oos, y_oos, reference=lgb_train, categorical_feature=cat_cols)
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=nrounds,
                    valid_sets=lgb_eval,
                    verbose_eval=True,
                    categorical_feature=cat_cols,
                    early_stopping_rounds=10)
    pred_final += gbm.predict(X_test, num_iteration=gbm.best_iteration)
    pred_oos = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
    df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
    train_stack.loc[test_index] = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    j += 1


df_results
np.mean(df_results['logloss'])
## .2806

pred_final /= nfolds
test_stack = pd.concat([pd.DataFrame(test_id, columns = ['claim_id']), pd.DataFrame(pred_final)], axis=1)
train_stack = pd.concat([pd.DataFrame(train_id, columns = ['claim_id']), train_stack], axis=1)

#train_stack.to_csv('train_stack_lgb_1.csv.gz', index = False, compression='gzip')
#test_stack.to_csv('test_stack_lgb_1.csv.gz', index = False, compression='gzip')

###Level 2###

num_cols = [f for f in train.columns if 'cat' not in f]

train_lvl2 = pd.merge(train[num_cols], train_stack, on='claim_id', how='inner', sort=False)
test_lvl2 = pd.merge(test[num_cols], test_stack, on='claim_id', how='inner', sort=False)

X = train_lvl2.drop(['claim_id', 'loss', 'target'], axis=1)
X_test = test_lvl2.drop(['claim_id'], axis=1)

params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 21,
          'colsample_bytree': .1,
          'min_child_weight': 5,
          'subsample': 1,
          'max_depth': 20,
          'alpha': 3,
          'gamma': 5,
          'lambda': 3,
          'nthread': 16,
          'silent': 1,
          'eta': .02,
          'verbose_eval': True
        }

dtrain = xgb.DMatrix(X.values, label=y_train)
xgtest = xgb.DMatrix(X_test.values)

bst_cv = xgb.cv(params, dtrain, num_boost_round=10000,
                early_stopping_rounds=25,
                nfold=5,
                verbose_eval=50)

print(bst_cv['test-mlogloss-mean'].values[-1])
nrounds = round(bst_cv.shape[0]/.8)

bst = xgb.train(params, dtrain, num_boost_round = nrounds,
                verbose_eval=50)

pred_final = bst.predict(xgtest)

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_final)], axis=1)

df.to_csv('lgblvl1_xgblvl2_data12_1.csv.gz', index = False, compression='gzip')
