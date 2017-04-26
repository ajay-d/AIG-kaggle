import numpy as np
np.random.seed(444)

import os
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

os.chdir("C:\\Users\\adeonari\\Downloads\\AIG\\data")
os.getcwd()

train = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_12.csv.gz', compression="gzip")

nfolds = 5
kf = KFold(nfolds, shuffle=True)

y_train = train['target'].ravel()
X = train.drop(['claim_id', 'loss', 'target'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

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

#To store stacked values
train_stack = pd.DataFrame(np.zeros(shape=(train.shape[0], 21)))
test_stack = pd.DataFrame(np.zeros(shape=(test.shape[0], 21)))

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

nrounds = 10000
pred_final = np.zeros(shape=(X_test.shape[0], 21))
j = 1
xgtest = xgb.DMatrix(X_test)
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.loc[train_index], X.loc[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    pred_oos = np.zeros(shape=(y_oos.shape[0], 21))
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, nrounds, watchlist, 
                    early_stopping_rounds=5, evals_result=evals_result,
                    verbose_eval=50)
    print("Best itr", bst.best_iteration)
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
    df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
    train_stack.loc[test_index] = bst.predict(dtest, ntree_limit=bst.best_iteration)
    j += 1


df_results
np.mean(df_results['logloss'])
## .2806

pred_final /= nfolds
test_stack = pd.concat([pd.DataFrame(test_id, columns = ['claim_id']), pd.DataFrame(pred_final)], axis=1)
train_stack = pd.concat([pd.DataFrame(train_id, columns = ['claim_id']), train_stack], axis=1)

train_stack.to_csv('train_stack_xgb_1.csv.gz', index = False, compression='gzip')
test_stack.to_csv('test_stack_xgb_1.csv.gz', index = False, compression='gzip')