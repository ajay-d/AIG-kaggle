import numpy as np
np.random.seed(4444)

import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = pd.read_csv('train_recode_5.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_5.csv.gz', compression="gzip")
train.shape
test.shape

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in all_data.columns[1:]:
    s = (all_data[f] - all_data[f].mean()) / all_data[f].std()
    frames = [df_normal, s]
    df_normal = pd.concat(frames, axis=1)

train_normal = pd.merge(pd.DataFrame(train['claim_id']), df_normal, on='claim_id', how='inner', sort=False)
test_normal = pd.merge(pd.DataFrame(test['claim_id']), df_normal, on='claim_id', how='inner', sort=False)

y_train = train['target'].ravel()
X = train_normal.drop(['claim_id'], axis=1)
X_test = test_normal.drop(['claim_id'], axis=1)

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
          'nthread': 64,
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

df.to_csv('gbm_m4_data5_norm.csv.gz', index = False, compression='gzip')