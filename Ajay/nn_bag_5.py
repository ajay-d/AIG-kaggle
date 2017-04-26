import numpy as np
np.random.seed(123)

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU

train = pd.read_csv('train_recode_12.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_12.csv.gz', compression="gzip")
train.shape
##(308386, 312)
test.shape

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index
all_data.head()

cat_cols = [f for f in all_data.columns if 'cat' in f]
num_cols = [f for f in all_data.columns if 'cat' not in f]
num_cols.remove('claim_id')
binary_cols = [f for f in all_data.columns if 'any' in f]

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in all_data.columns:
    if 'cat' in f:
        #Try 100, 200
        all_data.ix[all_data[f] > 100, f] = 100
        d = pd.get_dummies(all_data[f])
        frames = [df_normal, d]
        df_normal = pd.concat(frames, axis=1)

ms = MinMaxScaler(feature_range=(-1, 1)) 
all_data[num_cols] = ms.fit_transform(all_data[num_cols])

len(df_normal.columns)
df_normal.head()
all_data[num_cols].head()
all_data[binary_cols].head()

#combine categorical and numeric features
df_normal = pd.concat([df_normal, all_data[num_cols]], axis=1)

train_norm = pd.merge(pd.DataFrame(train['claim_id']), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test['claim_id']), df_normal, on='claim_id', how='inner', sort=False)

train_norm.shape
##(308386, 6797) @ 200
##(308386, 3919) @ 100
##(308386, 2408) @ 50

y_train = train['target'].ravel()
X = train_norm.drop(['claim_id'], axis=1)
X_test = test_norm.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

nfolds = 5
kf = KFold(nfolds, shuffle=True)
labels = np.arange(21)

##To store all CV results
df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(1000, input_dim=X.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.40))
        model.add(Dense(1000, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(4000, input_dim=X.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.40))
        model.add(Dense(2000, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(2000, input_dim=X.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.50))
        model.add(Dense(1000, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.25))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model
    if m==4:
        model = Sequential()
        model.add(Dense(4000, input_dim=X.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.40))
        model.add(Dense(1000, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dropout(0.20))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model
    if m==5:
        model = Sequential()
        model.add(Dense(2000, input_dim=X.shape[1], init='glorot_normal')) #800
        model.add(Activation('relu'))
        model.add(Dropout(0.6))
        model.add(Dense(100, init='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.3))
        model.add(Dense(50, init='glorot_uniform'))
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        model.add(Dense(21, init='glorot_uniform', activation='softmax'))
        return model

def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(shape=(X_test.shape[0], 21))
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos =np.zeros(shape=(y_oos.shape[0], 21))
        checkpoint = ModelCheckpoint('Checkpoint-Fold-%d.hdf5' % (j), save_best_only=True, verbose=1)
        early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1) 
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='adam', loss='categorical_crossentropy')
        nn_model.fit(X_model, y_model, batch_size=10000, nb_epoch=20, 
                     validation_data = (X_oos, y_oos), shuffle=True,
                     callbacks=[checkpoint, early_stop])
        nn_model.load_weights('Checkpoint-Fold-%d.hdf5' % (j))
        pred_final += nn_model.predict_proba(X_test.values, batch_size=10000)
        pred_oos = nn_model.predict_proba(X_oos)
        print("Fold", j, "logloss oos: ", log_loss(y_oos, pred_oos))
        df_results.loc[(df_results['fold']==j), 'logloss'] = log_loss(y_oos, pred_oos)
        j += 1
    pred_final /= nfolds
    return pred_final

pred_1 = model_cv(X.values, to_categorical(y_train), 3)
df_results
np.mean(df_results['logloss'])
#4 .2497


df = pd.DataFrame(test_id, columns = ['claim_id'])
df = pd.concat([df, pd.DataFrame(pred_1)], axis=1)

df.to_csv('nn_bag12_m5.csv.gz', index = False, compression='gzip')

