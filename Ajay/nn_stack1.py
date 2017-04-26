import numpy as np
np.random.seed(666)

import os
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, PReLU

os.chdir("C:\\Users\\adeonari\\Downloads\\AIG\\data")
os.getcwd()

train = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_12.csv.gz', compression="gzip")

cat_cols = [f for f in train.columns if 'cat' in f]
#cat_cols.insert(0,"claim_id")

nfolds = 4
kf = KFold(nfolds, shuffle=True)

y_train = train['target'].ravel()
X = train[cat_cols]
X_test = test[cat_cols]

train_id = train['claim_id'].values
test_id = test['claim_id'].values

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'min_gain_to_split': 0,
    'learning_rate': 0.02,
    #'num_leaves': 127,
    'num_leaves': 1023,
    'bagging_freq': 5,
    'num_threads': 6,
    #'max_bin' : 255,
    'max_bin' : 511,
    'num_class': 21,
    'verbose': 1
}

#To store stacked values
#train_stack = pd.DataFrame(train_id, columns = ['claim_id'])
#test_stack = pd.DataFrame(test_id, columns = ['claim_id'])
#train_stack = pd.concat([train_stack, pd.DataFrame(np.zeros(shape=(train_stack.shape[0], 21)))], axis=1)
#test_stack = pd.concat([test_stack, pd.DataFrame(np.zeros(shape=(test_stack.shape[0], 21)))], axis=1)
train_stack = pd.DataFrame(np.zeros(shape=(train.shape[0], 21)))
test_stack = pd.DataFrame(np.zeros(shape=(test.shape[0], 21)))

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

nrounds = 500
pred_final = np.zeros(shape=(X_test.shape[0], 21))
j = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.loc[train_index], X.loc[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    pred_oos = np.zeros(shape=(y_oos.shape[0], 21))
    lgb_train = lgb.Dataset(X_model, y_model, categorical_feature=cat_cols)
    lgb_eval = lgb.Dataset(X_oos, y_oos, reference=lgb_train, categorical_feature=cat_cols)
    gbm = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=nrounds,
                    valid_sets=lgb_eval,
                    verbose_eval=True,
                    categorical_feature=cat_cols,
                    early_stopping_rounds=10)
    pred_final += gbm.predict(X_test, num_iteration=gbm.best_iteration)
    pred_oos = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
    df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
    train_stack.loc[test_index] = gbm.predict(X_oos, num_iteration=gbm.best_iteration)
    j += 1


df_results
np.mean(df_results['logloss'])
## .2806

pred_final /= nfolds
test_stack = pd.concat([pd.DataFrame(test_id, columns = ['claim_id']), pd.DataFrame(pred_final)], axis=1)
train_stack = pd.concat([pd.DataFrame(train_id, columns = ['claim_id']), train_stack], axis=1)

#train_stack.to_csv('train_stack_lgb_1.csv.gz', index = False, compression='gzip')
#test_stack.to_csv('test_stack_lgb_1.csv.gz', index = False, compression='gzip')

###Level 2###

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

cat_cols = [f for f in train.columns if 'cat' in f]
y_train = train['target'].ravel()

train_stack = pd.read_csv('train_stack_lgb_1.csv.gz', compression="gzip")
test_stack = pd.read_csv('test_stack_lgb_1.csv.gz', compression="gzip")

#train_stack.drop(['claim_id.1', 'claim_id.2'], inplace=True, axis=1)

num_cols = [f for f in test.columns if 'cat' not in f]

train_lvl2 = pd.merge(train[num_cols], train_stack, on='claim_id', how='inner', sort=False)
test_lvl2 = pd.merge(test[num_cols], test_stack, on='claim_id', how='inner', sort=False)

train_lvl2['train_flag'] = 1
test_lvl2['train_flag'] = 0
ttdata = train_lvl2.append(test_lvl2)
num_cols.remove('claim_id')

ms = MinMaxScaler()
ttdata[num_cols] = ms.fit_transform(ttdata[num_cols])

train_lvl2 = ttdata[ttdata['train_flag']==1]
test_lvl2 = ttdata[ttdata['train_flag']==0]
train_lvl2.drop('train_flag', inplace = True, axis=1)
test_lvl2.drop('train_flag', inplace = True, axis=1)

X = train_lvl2.drop(['claim_id'], axis=1)
X_test = test_lvl2.drop(['claim_id'], axis=1)

X.shape
X_test.shape

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(500, input_dim=X.shape[1], init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, init='glorot_normal'))
        model.add(Activation('relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(500, init='glorot_normal'))
        model.add(Activation('relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(21, init='glorot_normal', activation='softmax'))
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(500, input_dim=X.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(500, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(500, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(750, input_dim=X.shape[1], init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, init='glorot_normal'))
        model.add(Activation('relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(500, init='glorot_normal'))
        model.add(Activation('relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(21, init='glorot_normal', activation='softmax'))
        return model
    if m==4:
        model = Sequential()
        model.add(Dense(500, input_dim=X.shape[1], init='glorot_normal'))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(500, init='glorot_normal'))
        model.add(Activation('relu')) 
        model.add(Dropout(0.2)) 
        model.add(Dense(21, init='glorot_normal', activation='softmax'))
        return model


def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(shape=(X_test.shape[0], 21))
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(shape=(y_oos.shape[0], 21))
        checkpoint = ModelCheckpoint('Checkpoint-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1) 
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, 
                                      verbose=2, mode='min',cooldown=5, min_lr=0.00001)
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='adam', loss='categorical_crossentropy')
        nn_model.fit(X_model, y_model, batch_size=10000, nb_epoch=100, 
                     validation_data = (X_oos, y_oos), shuffle=True, verbose=2,
                     callbacks=[checkpoint, early_stop])
        nn_model.load_weights('Checkpoint-Fold-%d.hdf5' % (j))
        pred_final += nn_model.predict_proba(X_test.values, batch_size=10000)
        pred_oos = nn_model.predict_proba(X_oos)
        print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
        df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
        j += 1
    pred_final /= nfolds
    return pred_final

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_1 = model_cv(X.values, to_categorical(y_train), 1)
df_results
df_1 = np.mean(df_results['logloss'])
#1: .2436 no reduce lr
#2: .2434 no reduce lr
#3: .2436 no reduce lr
#4: .2446 no reduce lr
df_1

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_2 = model_cv(X.values, to_categorical(y_train), 2)
df_results
df_2 = np.mean(df_results['logloss'])
df_2

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_3 = model_cv(X.values, to_categorical(y_train), 3)
df_results
df_3 = np.mean(df_results['logloss'])
df_3

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_4 = model_cv(X.values, to_categorical(y_train), 4)
df_results
df_4 = np.mean(df_results['logloss'])

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_1)], axis=1)
df.to_csv('lgblvl1_nnlvl2_data12_1.csv.gz', index = False, compression='gzip')

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_2)], axis=1)
df.to_csv('lgblvl1_nnlvl2_data12_2.csv.gz', index = False, compression='gzip')

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_3)], axis=1)
df.to_csv('lgblvl1_nnlvl2_data12_3.csv.gz', index = False, compression='gzip')

df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_4)], axis=1)
df.to_csv('lgblvl1_nnlvl2_data12_4.csv.gz', index = False, compression='gzip')



