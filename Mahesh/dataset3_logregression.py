import datetime
import numpy as np
import pandas as pd
import sys
import platform
import time

from sklearn.preprocessing import LabelEncoder, MinMaxScaler, scale
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

print("Read datasets...")
train_foo = pd.read_pickle("feat3_train_nn.pkl")
test_foo =  pd.read_pickle("feat3_test_nn.pkl")
labels = pd.read_pickle("labels_foo.pkl")
print("Done with datasets...")

train_foo['imparmt_pct'] = np.log(1+ train_foo['imparmt_pct'])

test_foo['imparmt_pct'] = np.log(1+ test_foo['imparmt_pct'])

train_foo['train_flag'] = 1
test_foo['train_flag'] = 0

ttdata= train_foo.append(test_foo)
col2scale = ['econ_unemployment_py', 'diagnosis_icd9_cd',  'aww1',
       'imparmt_pct', 'occ_desc1', 'diff_repo_loss', 'diff_impmt_loss',
       'diff_mmi_loss', 'diff_suit_loss', 'diff_paid_repo',
       'diff_paid_impmt', 'diff_paid_mmi', 'diff_paid_death',
       'diff_paid_return', 'diff_paid_suit', 'age_claimaint',
       'target_prev','target_median3', 'lp1', 'i_prev',
       'm_prev', 'tpd_amt', 'l_prev', 'i_prev_avg3', 'm_prev_avg3'] 
for x in col2scale:
    ttdata[x] = scale(ttdata[x])

ttdata = pd.get_dummies(ttdata, columns = ['cas_aia_cds_1_2', 'clm_aia_cds_1_2', 'clmnt_gender',
       'clm_aia_cds_3_4', 'initl_trtmt_cd',
       'catas_or_jntcvg_cd', 'state',  'suit_matter_type',
       'law_limit_tt', 'law_limit_pt', 'law_cola', 'law_offsets',
       'law_ib_scheduled','cutoffyr_flag', 'slice'])

train_foo = ttdata[ttdata['train_flag']==1]
test_foo = ttdata[ttdata['train_flag']==0]
train_foo.drop('train_flag', inplace = True, axis=1)
test_foo.drop('train_flag', inplace = True, axis=1)

# LOGISTIC REGRESSION ON DATA - WITH DUMMIES, NO SCALE

clf = LogisticRegression(penalty='l2', max_iter=50, multi_class='multinomial', solver='sag',
                                        verbose=2, n_jobs=-1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)


scores = [] 

dataset_blend_train_lr = pd.DataFrame(np.zeros((train_foo.shape[0], 21)))
dataset_blend_test_lr = pd.DataFrame(np.zeros((test_foo.shape[0], 21)))
    
i=0
a={}

for train_index, test_index in skf.split(train_foo, labels):
        
    print ("Fold ->", i+1)
    X_train, X_test = train_foo.iloc[train_index], train_foo.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        
    clf.fit(X_train, y_train)
        
    y_val = clf.predict_proba(X_test)
    
    score = log_loss(y_test, y_val)
    print (score)
    scores.append(score)

    dataset_blend_train_lr.iloc[test_index, 0:21] = y_val
        
    a[i] = pd.DataFrame(clf.predict_proba(test_foo)) 
        
    i+=1
        
dataset_blend_test_lr = a[0].copy()
for x in dataset_blend_test_lr.columns:
    dataset_blend_test_lr[x] = (a[0][x]+a[1][x]+a[2][x]+a[3][x]+a[4][x])/5

scores_mean =  np.mean(scores)
print (scores_mean)

dataset_blend_train_lr.to_pickle('FEAT3_LVL1_LR_TRAIN_' + str(scores_mean) + '_.pkl')
dataset_blend_test_lr.to_pickle('FEAT3_LVL1_LR_TEST_'+ str(scores_mean) + '_.pkl')    
    
print ("Done with LOGISTIC REGRESSION------>")