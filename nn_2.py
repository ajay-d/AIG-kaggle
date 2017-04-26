
import numpy as np
np.random.seed(666)

import os
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.advanced_activations import LeakyReLU, PReLU

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

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(1000, input_dim=X.shape[1], init='glorot_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(500, init='glorot_normal'))
        model.add(PReLU())
        model.add(Dropout(0.1))
        model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(1000, input_dim=X.shape[1], init='glorot_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1000, init='glorot_normal'))
        model.add(LeakyReLU())
        model.add(Dense(1000, init='glorot_normal'))
        model.add(LeakyReLU())        
        model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(1000, input_dim=X.shape[1], init='glorot_normal'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1000, init='glorot_normal'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1000, init='glorot_normal'))
        model.add(LeakyReLU())
        model.add(Dropout(0.2))        
        model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
        return model
    if m==4:
        model = Sequential()
        model.add(Dense(1000, input_dim=X.shape[1], init='glorot_normal'))
        model.add(PReLU())
        model.add(Dense(500, init='glorot_normal'))
        model.add(PReLU())
        model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
        return model
    if m==5:
        model = Sequential()
        model.add(Dense(500, input_dim=X.shape[1], init='glorot_normal'))
        model.add(PReLU())
        model.add(Dense(500, init='glorot_normal'))
        model.add(PReLU())
        model.add(Dense(500, init='glorot_normal'))
        model.add(PReLU())        
        model.add(Dense(1, init='glorot_normal', activation='sigmoid'))
        return model


def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(shape=(X_test.shape[0], 1))
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(shape=(y_oos.shape[0], 1))
        checkpoint = ModelCheckpoint('Checkpoint-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1) 
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, 
                                      verbose=2, mode='min',cooldown=5, min_lr=0.00001)
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='adam', loss='binary_crossentropy')
        nn_model.fit(X_model, y_model, batch_size=500, nb_epoch=100, 
                     validation_data = (X_oos, y_oos), shuffle=True, verbose=2,
                     callbacks=[checkpoint, early_stop])
        nn_model.load_weights('Checkpoint-Fold-%d.hdf5' % (j))
        pred_final += nn_model.predict_proba(X_test.values, batch_size=500)
        pred_oos = nn_model.predict_proba(X_oos)
        print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
        df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
        j += 1
    pred_final /= nfolds
    return pred_final

df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_1 = model_cv(X.values, y_train, 1)
df_results
df_1 = np.mean(df_results['logloss'])
#.6915

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_2 = model_cv(X.values, y_train, 2)
df_results
df_2 = np.mean(df_results['logloss'])
#.69115

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_3 = model_cv(X.values, y_train, 3)
df_results
df_3 = np.mean(df_results['logloss'])
#.69115

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_4 = model_cv(X.values, y_train, 4)
df_results
df_4 = np.mean(df_results['logloss'])
#.69148

df_results['logloss'] = np.zeros(df_results.shape[0])
pred_5 = model_cv(X.values, y_train, 5)
df_results
df_5 = np.mean(df_results['logloss'])
#.69139

df = pd.DataFrame(test_id, columns = ['t_id'])
df['nn_1'] = pred_1
df['nn_2'] = pred_2
df['nn_3'] = pred_3
df['nn_4'] = pred_4
df['nn_5'] = pred_5

df.to_csv('nn_2.csv.gz', index = False, compression='gzip')
