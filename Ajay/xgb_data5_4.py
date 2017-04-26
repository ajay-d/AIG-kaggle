import numpy as np
np.random.seed(444)

import feather
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = pd.read_csv('train_recode_5.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_5.csv.gz', compression="gzip")
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

df.to_csv('gbm_m4_data5.csv.gz', index = False, compression='gzip')