
from __future__ import division
version = 'xgb_10_2_models_wshi_02_all'

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pylab as plt
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.cross_validation import KFold, train_test_split
import operator
from itertools import combinations

def loo(dsn, varlist, y, split, r_k=.3):
    
    #median of target as complement
    y0 = np.mean(y[split])
    
    # table for all records
    df1 = dsn[varlist]
    df1['y'] = y
    
    # table for training only
    df2 = df1[split]
    df2['cnt'] = 1.0
    
    # calculate grouped total y and count
    grouped = df2.groupby(varlist).sum().add_prefix('sum_')
    
    # merge back to whole table
    df1 = pd.merge(df1, grouped, left_on = varlist, right_index = True, how = 'left')
    
    # leave one out
    df1['sum_cnt'] = np.where(split, df1['sum_cnt']-1, df1['sum_cnt'])
    df1['sum_y'] = np.where(split, df1['sum_y']-df1['y'], df1['sum_y'])
    
    # calculated transformed cats based on credibilty
    cred_k = 2
    
    df1['loo_y'] = (df1['sum_y'] + y0*cred_k)*1.0 / (df1['sum_cnt'] + cred_k)
    df1['loo_y'][df1['loo_y'].isnull()] = y0
    
    # add noise to pretent overfitting
    df1['loo_y'][split] = df1['loo_y'][split]*(1+(np.random.uniform(0,1,sum(split))-0.5)*r_k)
    return df1['loo_y']

split_id = pd.read_csv('./split.csv')
train = pd.read_csv('/mnt/aigdata/OriginalData/train.csv', encoding='latin1')

train_2015 = train[['claim_id', 'target']][train.paid_year==2015]
train_2015.shape
    
for sl in range(1,11,1):

    train_test_p2015 = pd.read_csv('/mnt/junwang/project6/data/R_10_model_all/train.test.model.{0}.all.cat.csv'.format(sl))
    train_test_p2015 = pd.merge(train_test_p2015, split_id, how='left', on=['claim_id'])
    train_test_p2015.split[train_test_p2015.split.isnull()] = -1
                           
    train_test_p2015.drop(['target'],axis=1,inplace=True)
    train_test_p2015 = pd.merge(train_test_p2015, train_2015, on = ['claim_id'], how='left')
                               
    train_val = train_test_p2015[train_test_p2015.split.isin([0,1,2,3])].reset_index(drop=True)
    train_val.drop(['split'],axis=1,inplace=True)
    hold_test = train_test_p2015[train_test_p2015.split.isin([-1])].reset_index(drop=True)

    pred_test_tot = np.zeros((train_test_p2015[train_test_p2015.split==-1].shape[0], 21))
    
    del(train_test_p2015)

    nfolds = 10
    
    kf = KFold(train_val.shape[0], n_folds=nfolds)
    for fold, (train_index, val_index) in enumerate(kf):
        print('Fold {0}'.format(fold+1))
        tr, val = train_val.ix[train_index], train_val.ix[val_index]

        tr['split'] = 1
        val['split'] = 2
        tv = tr.append(val)

        train_test_p2015 = (tv.append(hold_test)).reset_index(drop=True)
        
        cats = train_test_p2015.dtypes[train_test_p2015.dtypes=='object'].index
        for var in cats:
            train_test_p2015[var] = np.where(train_test_p2015[var].isnull(), '_M', train_test_p2015[var])
            train_test_p2015[var] = loo(train_test_p2015,[var],np.log(train_test_p2015.target+1),train_test_p2015.split==1,r_k=0.3)
          
        x_train = train_test_p2015[train_test_p2015.split==1].reset_index(drop=True)
        x_val = train_test_p2015[train_test_p2015.split==2].reset_index(drop=True)
        x_test = train_test_p2015[train_test_p2015.split==-1].reset_index(drop=True)
        
        id_test = x_test['claim_id'].values
    
        bins = [-np.inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000,\
                   94000, 133000, 189000, 268000, 381000, 540000, np.inf]
        bins_name = range(21)
        
        y_train = pd.cut(x_train.target, bins, labels=bins_name)
        y_val = pd.cut(x_val.target, bins, labels=bins_name)
                  
        train_target = x_train[['target']]
        val_target = x_val[['target']]
        
        x_train.drop(['claim_id', 'target', 'split'], 1, inplace=True)
        x_val.drop(['claim_id', 'target', 'split'], 1, inplace=True)
        x_test.drop(['claim_id', 'target', 'split'], 1, inplace=True)
        
        
        bins = [-np.inf, 0, np.inf]
        bins_name = range(2)
        
        y1_train = pd.cut(train_target.target, bins, labels=bins_name)
        y1_val = pd.cut(val_target.target, bins, labels=bins_name)
        y1_train.value_counts()
        
        d1train = xgb.DMatrix(x_train, label=y1_train, missing=np.nan)
        d1val = xgb.DMatrix(x_val, label=y1_val, missing=np.nan)
        dtest = xgb.DMatrix(x_test, missing=np.nan)
        
        pred1_test = np.zeros((x_test.shape[0], 2))
        
        params = {
            'objective':'multi:softprob',
            'num_class': 2,
            'eval_metric': 'mlogloss',
            'min_child_weight': 64,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'max_depth': 9,
            'subsample': 0.9,
            #'alpha': 0.5,
            #'lambda':0.5,
            #'gamma': 1,
            'silent': 1,
            'seed': 6688
        }
        
        watchlist = [(d1train, 'train'), (d1val, 'val')]      
        clf1 = xgb.train(params, d1train, 10000, watchlist, early_stopping_rounds=20, verbose_eval=False)
        
        pred1_test = clf1.predict(dtest,ntree_limit=clf1.best_ntree_limit).reshape(pred1_test.shape)
        
        
        bins = [0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000,\
                   94000, 133000, 189000, 268000, 381000, 540000, np.inf]
        bins_name = range(20)
        
        y2_train = pd.cut(train_target.target[train_target.target>0], bins, labels=bins_name)
        y2_val = pd.cut(val_target.target[val_target.target>0], bins, labels=bins_name)
        y2_train.value_counts()
        
        x2_train = x_train[train_target.target>0]
        x2_val = x_val[val_target.target>0]
        print(x2_train.shape, x2_val.shape, y2_train.shape, y2_val.shape)
        
        d2train = xgb.DMatrix(x2_train, label=y2_train, missing=np.nan)
        d2val = xgb.DMatrix(x2_val, label=y2_val, missing=np.nan)
            
        pred2_test = np.zeros((x_test.shape[0], 20))
        
        
        params = {
            'objective':'multi:softprob',
            'num_class': 20,
            'eval_metric': 'mlogloss',
            'min_child_weight': 61,
            'eta': 0.1,
            'colsample_bytree': 0.5,
            'max_depth': 5,
            'subsample': 0.9,
            #'alpha': 1,
            #'lambda':0,
            #'gamma': 3,
            'silent': 1,
            'seed': 6688
        }
        
        watchlist = [(d2train, 'train'), (d2val, 'val')]      
        clf2 = xgb.train(params, d2train, 10000, watchlist, early_stopping_rounds=20, verbose_eval=False)
        
        pred2_test = clf2.predict(dtest,ntree_limit=clf2.best_ntree_limit).reshape(pred2_test.shape)
        
        pred_test = np.zeros((x_test.shape[0], 21))
        
        pred_test[:,0] = pred1_test[:,0]
        
        for i in range(20):
            pred_test[:,i+1] = pred1_test[:,1]*pred2_test[:,i]
        
        pred_test_tot += pred_test
        
    '''
    combine cross validation results
    '''

    pred_test_tot = pred_test_tot/nfolds
    
    if sl == 1:
        pred_test_combine = np.c_[id_test, pred_test_tot]
    else:
        pred_test_combine = np.r_[pred_test_combine, np.c_[id_test, pred_test_tot]]


header = ['claim_id','Bucket_1','Bucket_2','Bucket_3','Bucket_4','Bucket_5','Bucket_6','Bucket_7','Bucket_8','Bucket_9','Bucket_10','Bucket_11','Bucket_12','Bucket_13','Bucket_14','Bucket_15','Bucket_16','Bucket_17','Bucket_18','Bucket_19','Bucket_20','Bucket_21']

subm = pd.DataFrame(pred_test_combine, columns=header)
subm.sort_values(by=['claim_id'], axis=0,inplace=True)
subm.claim_id = subm.claim_id.astype(int)
subm.to_csv('./submission/'+ version + '.csv', index=False)        

print("Done")



    
