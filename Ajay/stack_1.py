import numpy as np
np.random.seed(888)

import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

train = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_8.csv.gz', compression="gzip")
train.shape
test.shape


y_train = train['target'].ravel()
X = train.drop(['claim_id', 'loss', 'target'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

nfolds = 5
kf = KFold(nfolds, shuffle=True)

boost_1 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, splitter='best'), n_estimators = 250)
boost_2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2, splitter='random'), n_estimators = 250)

etr_1 = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='gini')
etr_2 = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy')
etr_3 = ExtraTreesClassifier(n_estimators=10, max_depth=5, criterion='gini')
etr_4 = ExtraTreesClassifier(n_estimators=10, max_depth=5, criterion='entropy')

boost_3 = AdaBoostClassifier(etr_3, n_estimators = 250, learning_rate=.5)
boost_4 = AdaBoostClassifier(etr_4, n_estimators = 250, learning_rate=.5)

rf_1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='gini')
rf_2 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, criterion='entropy')

pred_final = np.zeros(shape=(X_test.shape[0], 21))
models = [rf_1, rf_2, etr_1, etr_2, boost_1, boost_2, boost_3, boost_4]

#To store stacked values
train_stack = pd.DataFrame(train_id, columns = ['claim_id'])
test_stack = pd.DataFrame(test_id, columns = ['claim_id'])
labels = np.arange(21)

blend_cols_tr = pd.DataFrame(np.zeros(shape=(train_stack.shape[0], 21 * len(models))))
blend_cols_te = pd.DataFrame(np.zeros(shape=(test_stack.shape[0], 21 * len(models))))
train_stack = pd.concat([train_stack, blend_cols_tr], axis=1)
test_stack = pd.concat([test_stack, blend_cols_te], axis=1)

fold = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    mod = 1
    for i in models:
        i.fit(X_model, y_model)
        start_col = (mod-1)*21 + 1
        train_stack.ix[test_index, start_col:(start_col+21)] = i.predict_proba(X_oos)
        test_stack.ix[:, start_col:(start_col+21)] += i.predict_proba(X_test.values)
        pred_oos = i.predict_proba(X_oos)
        print("Fold", fold, "model", mod, "logloss: ", log_loss(y_oos, pred_oos, labels=labels))
        mod += 1
    fold += 1

#average columns folds for test set   
test_stack.ix[:, 1:] /= nfolds

train_stack.to_csv('train_stack8_1.csv.gz', index = False, compression='gzip')
test_stack.to_csv('test_stack8_1.csv.gz', index = False, compression='gzip')