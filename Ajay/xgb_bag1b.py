import numpy as np
np.random.seed(555)

import pandas as pd
import xgboost as xgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold

train = pd.read_csv('train_recode_5.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_5.csv.gz', compression="gzip")
train.shape
test.shape

y_train = train['target'].ravel()

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index

cat_cols = ["adj", "clmnt_gender", "major_class_cd", "prexist_dsblty_in", "catas_or_jntcvg_cd",
            "suit_matter_type", "initl_trtmt_cd", "state",
            "occ_code", "sic_cd", "diagnosis_icd9_cd", "cas_aia_cds_1_2", "cas_aia_cds_3_4", "clm_aia_cds_1_2", "clm_aia_cds_3_4",
            "law_limit_tt", "law_limit_pt", "law_limit_pp", "law_cola", "law_offsets", "law_ib_scheduled",
            "dnb_spevnt_i", "dnb_rating", "dnb_comp_typ",
            "cat_1", "cat_2", "cat_3", "cat_4"]

           
df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in cat_cols:
    s = (all_data[f] - all_data[f].mean()) / all_data[f].std()
    frames = [df_normal, s]
    df_normal = pd.concat(frames, axis=1)

f_num = [f for f in all_data.columns if f not in cat_cols]

train_normal = pd.merge(train[f_num], df_normal, on='claim_id', how='inner', sort=False)
test_normal = pd.merge(test[f_num], df_normal, on='claim_id', how='inner', sort=False)

X = train_normal.drop(['claim_id'], axis=1)
X_test = test_normal.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 21,
          'colsample_bytree': .016,
          'min_child_weight': 8.84,
          'subsample': .727,
          'max_depth': 15,
          'alpha': 4.87,
          'gamma': 12.67,
          'lambda': 7.44,
          'nthread': 32,
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
                    early_stopping_rounds=25, evals_result=evals_result,
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

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(predictions)], axis=1)

df_results

df.to_csv('gbm_bag_m1_d15_norm.csv', index = False)
