import numpy as np
np.random.seed(333)

import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train_a = pd.read_csv('train_recode_6a.csv.gz', compression="gzip")
test_a = pd.read_csv('test_recode_6a.csv.gz', compression="gzip")
train_b = pd.read_csv('train_recode_6b.csv.gz', compression="gzip")
test_b = pd.read_csv('test_recode_6b.csv.gz', compression="gzip")

train = pd.concat([train_a, train_b], ignore_index=True)
test = pd.concat([test_a, test_b], ignore_index=True)

train.shape
test.shape

y_train = train['target'].ravel()
X = train.drop(['claim_id', 'loss', 'target'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 21,
          'colsample_bytree': .4,
          'min_child_weight': 5,
          'subsample': 1,
          'max_depth': 10,
          'alpha': 2.35,
          'gamma': 1.35,
          'lambda': 2.9,
          'nthread': 64,
          'silent': 1,
          'eta': .01,
          'verbose_eval': True
        }

nfolds = 5
kf = KFold(nfolds, shuffle=True)
pred_final = np.zeros(shape=(X_test.shape[0], 21))
i = 0
logloss_oos = 0

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

xgtest = xgb.DMatrix(X_test.values)
for train_index, test_index in kf.split(X.values):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    dtrain = xgb.DMatrix(X_model, label=y_model)
    dtest = xgb.DMatrix(X_oos, label=y_oos)
    watchlist  = [(dtrain,'train'), (dtest,'test_oos')]
    evals_result = {}
    bst = xgb.train(params, dtrain, 10000, watchlist, 
                    early_stopping_rounds=10, evals_result=evals_result,
                    verbose_eval=50)
    print("Best itr", bst.best_iteration)
    pred_oos = bst.predict(dtest, ntree_limit=bst.best_iteration)
    i += 1
    df_results.loc[(df_results['fold']==i), 'logloss'] = log_loss(y_oos, pred_oos)
    pred_final += bst.predict(xgtest, ntree_limit=bst.best_iteration)
    logloss_oos += log_loss(y_oos, pred_oos)
    print("Fold", i, "logloss oos: ", log_loss(y_oos, pred_oos))

predictions = pred_final/nfolds
##
print("Final Average LogLoss oos: ", logloss_oos/nfolds)

df_results

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(predictions)], axis=1)

df.to_csv('gbm_bag_m5b_d10.csv.gz', index = False, compression='gzip')
