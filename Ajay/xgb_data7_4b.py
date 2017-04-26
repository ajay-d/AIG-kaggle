import numpy as np
np.random.seed(7777)

import feather
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = feather.read_dataframe('train_0.feather')
test = feather.read_dataframe('test_0.feather')

for i in (np.arange(9)+1):
    tr = feather.read_dataframe('train_%d.feather' % i)
    te = feather.read_dataframe('test_%d.feather' % i)
    
    train = pd.concat([train, tr], ignore_index=True)
    test = pd.concat([test, te], ignore_index=True)

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
          'max_depth': 10,
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

df.to_csv('gbm_m4b_data7.csv.gz', index = False, compression='gzip')