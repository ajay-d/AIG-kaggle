import numpy as np
np.random.seed(44)

import feather
import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = pd.read_csv('train_recode_9a.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_9a.csv.gz', compression="gzip")
train.shape
test.shape

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index
all_data.head()

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in all_data.columns:
    if 'cat' in f:
        d = pd.get_dummies(all_data[f])
        frames = [df_normal, d]
        df_normal = pd.concat(frames, axis=1)

len(df_normal.columns)
df_normal.head()

num_cols = [f for f in all_data.columns if 'cat' not in f]
all_data[num_cols].head()

train_norm = pd.merge(pd.DataFrame(train[num_cols]), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test[num_cols]), df_normal, on='claim_id', how='inner', sort=False)

y_train = train['target'].ravel()
X = train_norm.drop(['claim_id'], axis=1)
X_test = test_norm.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 21,
          'colsample_bytree': .25,
          'min_child_weight': 5,
          'subsample': 1,
          'max_depth': 20,
          'alpha': 3,
          'gamma': 7,
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

df.to_csv('gbm_m4b_data9a_hot.csv.gz', index = False, compression='gzip')