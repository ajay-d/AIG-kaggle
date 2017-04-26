import numpy as np
np.random.seed(222)

import pandas as pd
import numpy as np
from sklearn import svm

train = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_8.csv.gz', compression="gzip")
train.shape
test.shape

y_train = train['target'].ravel()
all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index
all_data.head()

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in all_data.columns[1:]:
    s = (all_data[f] - all_data[f].mean()) / all_data[f].std()
    frames = [df_normal, s]
    df_normal = pd.concat(frames, axis=1)

train_norm = pd.merge(pd.DataFrame(train['claim_id']), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test['claim_id']), df_normal, on='claim_id', how='inner', sort=False)

#####Sample Data#####
train_svm = train_norm.sample(frac=0.1)
y_sample = pd.merge(train[['claim_id', 'target']], train_svm[['claim_id']], on='claim_id', how='inner', sort=False)
y_train = y_sample['target'].ravel()
#####################

#y_train = train['target'].ravel()
X = train_svm.drop(['claim_id'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train_svm['claim_id'].values
test_id = test['claim_id'].values

svc_rbf = svm.SVC(kernel='rbf', probability=True, decision_function_shape='ovr', cache_size=50000)
svc_lin = svm.SVC(kernel='poly')
nu_rbf = svm.NuSVC(kernel='rbf')
nu_lin = svm.NuSVC(kernel='poly')


svc_rbf.fit(X, y_train)
dec = svc_rbf.decision_function(X)
pred = svc_rbf.predict_proba(X)

