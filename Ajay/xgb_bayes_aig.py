import numpy as np
np.random.seed(999)

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from bayes_opt import BayesianOptimization

#Baysian hyperparameter optimization [https://github.com/fmfn/BayesianOptimization]

train = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
train.shape

y_train = train['target'].ravel()
X = train.drop(['claim_id', 'loss', 'target'], axis=1)


def xgb_evaluate(min_child_weight, colsample_bytree, max_depth, subsample, l, gamma, alpha):
    
    params['min_child_weight'] = int(round(min_child_weight))
    params['cosample_bytree'] = max(min(colsample_bytree, 1), 0)
    params['max_depth'] = int(round(max_depth))
    params['subsample'] = max(min(subsample, 1), 0)
    params['lambda'] = max(l, 0)
    params['gamma'] = max(gamma, 0)
    params['alpha'] = max(alpha, 0)
    
    cv_result = xgb.cv(params, dtrain, num_boost_round=num_rounds, nfold=5,
                       callbacks=[xgb.callback.early_stop(5)])
    return -cv_result['test-mlogloss-mean'].values[-1]


    
num_rounds = 500
num_iter = 10
init_points = 10

params = {"objective": "multi:softprob",
          "eval_metric": "mlogloss",
          "num_class": 21,
          'eta': 0.3,
          'silent': 1,
          'nthread': 12,
          'verbose_eval': True
    }
    
dtrain = xgb.DMatrix(X.values, label=y_train)

xgbBO = BayesianOptimization(xgb_evaluate, {'min_child_weight': (0, 10),
                                            'colsample_bytree': (0, 1),
                                            'max_depth': (10, 25),
                                            'subsample': (0, 1),
                                            'l': (0, 10),
                                            'gamma': (0, 15),
                                            'alpha': (0, 10),
                                            })


xgbBO.maximize(init_points=init_points, n_iter=num_iter)

print(xgbBO.res['max'])
