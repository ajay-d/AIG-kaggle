
import numpy as np
np.random.seed(777)

import os
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

os.chdir("C:\\Users\\adeonari\\Downloads\\numerai_datasets")
os.getcwd()

train = pd.read_csv("numerai_training_data.csv")
test = pd.read_csv("numerai_tournament_data.csv")

nfolds = 5
kf = KFold(nfolds, shuffle=True)

y_train = train['target'].ravel()
X = train.drop(['target'], axis=1)
X_test = test.drop(['t_id'], axis=1)

X.shape
X_test.shape

test_id = test['t_id'].values
train['t_id'] = np.arange(train.shape[0])+1
train_id = train['t_id'].values

nfolds = 5
kf = KFold(nfolds, shuffle=True)

knn_1a = KNeighborsClassifier(n_neighbors=2, n_jobs=-1, weights='uniform')
knn_1b = KNeighborsClassifier(n_neighbors=2, n_jobs=-1, weights='distance')
knn_2a = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='uniform')
knn_2b = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')
knn_3a = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, weights='uniform')
knn_3b = KNeighborsClassifier(n_neighbors=10, n_jobs=-1, weights='distance')
knn_4a = KNeighborsClassifier(n_neighbors=25, n_jobs=-1, weights='uniform')
knn_4b = KNeighborsClassifier(n_neighbors=25, n_jobs=-1, weights='distance')
knn_5a = KNeighborsClassifier(n_neighbors=50, n_jobs=-1, weights='uniform')
knn_5b = KNeighborsClassifier(n_neighbors=50, n_jobs=-1, weights='distance')
knn_6a = KNeighborsClassifier(n_neighbors=100, n_jobs=-1, weights='uniform')
knn_6b = KNeighborsClassifier(n_neighbors=100, n_jobs=-1, weights='distance')

pred_final = np.zeros(shape=(X_test.shape[0], 1))
models = [knn_1a, knn_1b, knn_2a, knn_2b, knn_3a, knn_3b, knn_4a, knn_4b, knn_5a, knn_5b, knn_6a, knn_6b]
#models = [knn_1a, knn_1b]

#To store stacked values
#probability of each class for each model
train_stack = pd.DataFrame(train_id, columns = ['t_id'])
test_stack = pd.DataFrame(test_id, columns = ['t_id'])

blend_cols_tr = pd.DataFrame(np.zeros(shape=(train_stack.shape[0], len(models))))
blend_cols_te = pd.DataFrame(np.zeros(shape=(test_stack.shape[0], len(models))))
train_stack = pd.concat([train_stack, blend_cols_tr], axis=1)
test_stack = pd.concat([test_stack, blend_cols_te], axis=1)

#To store stacked meta features
#sum of distances of 2/4 nearest neighbours 
train_stack_dist = pd.DataFrame(train_id, columns = ['t_id'])
test_stack_dist = pd.DataFrame(test_id, columns = ['t_id'])

blend_cols_tr_dist = pd.DataFrame(np.zeros(shape=(train_stack.shape[0], 2 * len(models))))
blend_cols_te_dist = pd.DataFrame(np.zeros(shape=(test_stack.shape[0], 2 * len(models))))
train_stack_dist = pd.concat([train_stack_dist, blend_cols_tr_dist], axis=1)
test_stack_dist = pd.concat([test_stack_dist, blend_cols_te_dist], axis=1)

fold = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    mod = 1
    print("Fold", fold, "model", mod)
    for i in models:
        i.fit(X_model, y_model)
        start_col = mod
        #train_stack.ix[test_index, start_col:(start_col+21)] = i.predict_proba(X_oos)
        train_stack.iloc[test_index, start_col] = i.predict_proba(X_oos)[:,0]
        #test_stack.ix[:, start_col:(start_col+21)] += i.predict_proba(X_test.values)
        test_stack.iloc[:, start_col] += i.predict_proba(X_test.values)[:,0]
        pred_oos = i.predict_proba(X_oos)
        print("Fold", fold, "model", mod, "logloss: ", log_loss(y_oos, pred_oos))
        start_col_dist = (mod-1)*2 + 1
        dist = i.kneighbors(X_oos, return_distance=True)
        dist_sum = np.sum(dist[0],1).reshape(X_oos.shape[0],1)
        dist_mean = np.mean(dist[0],1).reshape(X_oos.shape[0],1)
        dist_std = np.std(dist[0],1).reshape(X_oos.shape[0],1)
        cols = np.concatenate((dist_mean, dist_std), axis=1)
        train_stack_dist.ix[test_index, start_col_dist:(start_col_dist+2)] = cols
        
        dist = i.kneighbors(X_test.values, return_distance=True)
        dist_sum = np.sum(dist[0],1).reshape(X_test.values.shape[0],1)
        dist_mean = np.mean(dist[0],1).reshape(X_test.values.shape[0],1)
        dist_std = np.std(dist[0],1).reshape(X_test.values.shape[0],1)
        cols = np.concatenate((dist_mean, dist_std), axis=1)
        test_stack_dist.ix[:, start_col_dist:(start_col_dist+2)] += cols
        mod += 1
    fold += 1

#average columns folds for test set   
test_stack.ix[:, 1:] /= nfolds
test_stack_dist.ix[:, 1:] /= nfolds

train_stack_all = pd.merge(train_stack_dist, train_stack, on='t_id', how='inner', sort=False)
test_stack_all = pd.merge(test_stack_dist, test_stack, on='t_id', how='inner', sort=False)

train_stack_all.to_csv('train_knn_1.csv.gz', index = False, compression='gzip')
test_stack_all.to_csv('test_knn_1.csv.gz', index = False, compression='gzip')