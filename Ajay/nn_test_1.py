import numpy as np
np.random.seed(666)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

train = pd.read_csv('train_recode_5.csv.gz', compression="gzip")
train.shape

stack_cols = [f for f in train.columns if 'stack' in f]
train_nn = train[stack_cols]

y_train = train['target'].ravel()
all_data = train.drop(['loss', 'target'], axis=1)
all_data.index

cat_cols = ["adj", "clmnt_gender", "major_class_cd", "prexist_dsblty_in", "catas_or_jntcvg_cd",
            "suit_matter_type", "initl_trtmt_cd", "state",
            "occ_code", "sic_cd", "diagnosis_icd9_cd", "cas_aia_cds_1_2", "cas_aia_cds_3_4", "clm_aia_cds_1_2", "clm_aia_cds_3_4",
            "law_limit_tt", "law_limit_pt", "law_limit_pp", "law_cola", "law_offsets", "law_ib_scheduled",
            "dnb_spevnt_i", "dnb_rating", "dnb_comp_typ",
            "cat_1", "cat_2", "cat_3", "cat_4"]

#df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
#for f in cat_cols:
#    #all_data.ix[all_data[f] > 2500, f] = 2500
#    all_data.ix[all_data[f] > 1000, f] = 1000
#    d = pd.get_dummies(all_data[f])
#    frames = [df_normal, d]
#    df_normal = pd.concat(frames, axis=1)

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in cat_cols:
    s = (all_data[f] - all_data[f].mean()) / all_data[f].std()
    frames = [df_normal, s]
    df_normal = pd.concat(frames, axis=1)

len(all_data.columns)
len(cat_cols)
len(df_normal.columns)

f_num = [f for f in all_data.columns if f not in cat_cols]

train_nn = pd.merge(all_data[f_num], df_normal, on='claim_id', how='inner', sort=False)
train_nn.drop(['claim_id'], axis=1, inplace=True)

train_nn.shape
len(y_train)
to_categorical(y_train).shape

model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU()) 
model.add(Dropout(0.1))
model.add(Dense(10000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=10, shuffle=True, validation_split=.5)
#logloss .2553 @9 validate

model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dense(10000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=10, shuffle=True, validation_split=.5)
#logloss .2540 @12 validate -- go longer

model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(10000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.2))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=10, shuffle=True, validation_split=.5)
#logloss .2543 @10 validate

model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(10000, init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=10, shuffle=True, validation_split=.25)
#logloss .2608 @ 4 validate

model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(10000, init='glorot_uniform'))
model.add(LeakyReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=10, shuffle=True, validation_split=.5)
#logloss .2633 @10  validate

preds = model.predict(train_normal.values, batch_size=50000)
preds[0]
sum(preds[1])
preds[0:10,0]

##Using 2500
#(308386, 14354)
model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(10000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=25, shuffle=True, validation_split=.5)
#logloss .2549 @9  validate
          
model = Sequential()
model.add(Dense(20000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(20000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=10000, 
          nb_epoch=25, shuffle=True, validation_split=.5)