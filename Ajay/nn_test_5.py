import numpy as np
np.random.seed(44)

import pandas as pd
import tensorflow as tf
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cross_validation import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU, PReLU

train = pd.read_csv('train_recode_9a.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_9a.csv.gz', compression="gzip")
train.shape
test.shape

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index
all_data.head()

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in all_data.columns:
    if 'cat' in f:
        d = pd.get_dummies(all_data[f])
        frames = [df_normal, d]
        df_normal = pd.concat(frames, axis=1)

len(df_normal.columns)
df_normal.head()

num_cols = [f for f in all_data.columns if 'cat' not in f]
all_data[num_cols].head()

train_norm = pd.merge(pd.DataFrame(train[num_cols]), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test[num_cols]), df_normal, on='claim_id', how='inner', sort=False)

y_train = train['target'].ravel()
X = train_norm.drop(['claim_id'], axis=1)
X_test = test_norm.drop(['claim_id'], axis=1)

train_id = train['claim_id'].values
test_id = test['claim_id'].values

X, X_test, Y, Y_test = train_test_split(X.values, to_categorical(y_train), test_size=0.25, random_state=666)

fBestModel = 'best_model.h5' 
early_stop = EarlyStopping(monitor='val_loss', patience=4, verbose=1) 
best_model = ModelCheckpoint(fBestModel, verbose=1, save_best_only=True)

model = Sequential()
model.add(Dense(1000, input_dim=X.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dense(1000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dense(21, init='glorot_uniform', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, Y, validation_data = (X_test, Y_test), 
          nb_epoch=20, batch_size=10000, verbose=False,
          callbacks=[best_model, early_stop])
##8 @ 0.25278

model = Sequential()
model.add(Dense(1000, input_dim=X.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.40))
model.add(Dense(1000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.10))
model.add(Dense(21, init='glorot_uniform', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, Y, validation_data = (X_test, Y_test), 
          nb_epoch=20, batch_size=10000, verbose=False,
          callbacks=[best_model, early_stop])
##11 @ 0.25143

model = Sequential()
model.add(Dense(1000, input_dim=X.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.40))
model.add(Dense(800, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.20))
model.add(Dense(21, init='glorot_uniform', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, Y, validation_data = (X_test, Y_test), 
          nb_epoch=20, batch_size=10000, verbose=False,
          callbacks=[best_model, early_stop])
##11 @ 0.25188

model = Sequential()
model.add(Dense(1000, input_dim=X.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.40))
model.add(Dense(1000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.20))
model.add(Dense(21, init='glorot_uniform', activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy')
model.fit(X, Y, validation_data = (X_test, Y_test), 
          nb_epoch=20, batch_size=10000, verbose=False,
          callbacks=[best_model, early_stop])
##10 @ 0.25184
