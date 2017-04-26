import numpy as np
np.random.seed(888)

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

train = pd.read_csv('train_recode_5.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_5.csv.gz', compression="gzip")
train.shape
test.shape

y_train = train['target'].ravel()

all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index

cat_cols = ["adj", "clmnt_gender", "major_class_cd", "prexist_dsblty_in", "catas_or_jntcvg_cd",
            "suit_matter_type", "initl_trtmt_cd", "state",
            "occ_code", "sic_cd", "diagnosis_icd9_cd", "cas_aia_cds_1_2", "cas_aia_cds_3_4", "clm_aia_cds_1_2", "clm_aia_cds_3_4",
            "law_limit_tt", "law_limit_pt", "law_limit_pp", "law_cola", "law_offsets", "law_ib_scheduled",
            "dnb_spevnt_i", "dnb_rating", "dnb_comp_typ",
            "cat_1", "cat_2", "cat_3", "cat_4"]

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in cat_cols:
    all_data.ix[all_data[f] > 2500, f] = 2500
    d = pd.get_dummies(all_data[f])
    frames = [df_normal, d]
    df_normal = pd.concat(frames, axis=1)

len(all_data.columns)
len(cat_cols)
len(df_normal.columns)

f_num = [f for f in all_data.columns if f not in cat_cols]

train_nn = pd.merge(train[f_num], df_normal, on='claim_id', how='inner', sort=False)
test_nn = pd.merge(test[f_num], df_normal, on='claim_id', how='inner', sort=False)

#####Sample Data#####
#train_nn = train_nn.sample(frac=0.1)
#y_sample = pd.merge(train[['claim_id', 'target']], train_nn, on='claim_id', how='inner', sort=False)
#y_train = y_sample['target'].ravel()
#####################

train_nn.drop(['claim_id'], axis=1, inplace=True)
test_nn.drop(['claim_id'], axis=1, inplace=True)

all_data.shape
train_nn.shape
test_nn.shape
len(y_train)
to_categorical(y_train).shape

train_id = train['claim_id'].values
test_id = test['claim_id'].values

nfolds = 5
kf = KFold(nfolds, shuffle=True)

early_stopping = EarlyStopping(monitor='val_categorical_crossentropy', min_delta=0.0, patience=10, verbose=1)

df_results = pd.DataFrame(np.arange(0, nfolds, 1), columns=['fold'])
df_results['fold'] += 1
df_results['logloss'] = np.zeros(df_results.shape[0])

#To store stacked values
#probability of each class for each model
train_stack = pd.DataFrame(train_id, columns = ['claim_id'])
test_stack = pd.DataFrame(test_id, columns = ['claim_id'])
labels = np.arange(21)

def model_cv(X, y, model_n):
    fold = 1
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        start_col = (model_n-1)*21 + 1
        nn_model = create_model(model_n)
        nn_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
        checkpoint = ModelCheckpoint('Checkpoint-Fold-%d.hdf5' % (fold), save_best_only=True, verbose=1)
        nn_model.fit(X_model, to_categorical(y_model), batch_size=10000, nb_epoch=1, 
                     validation_data = (X_oos, to_categorical(y_oos)), shuffle=True,
                     callbacks=[early_stopping, checkpoint])
        #load checkpointed best model
        nn_model.load_weights('Checkpoint-Fold-%d.hdf5' % (fold))
        train_stack.ix[test_index, start_col:(start_col+21)] = nn_model.predict_proba(X_oos)
        test_stack.ix[:, start_col:(start_col+21)] += nn_model.predict_proba(test_nn.values)
        pred_oos = nn_model.predict_proba(X_oos)
        print("Fold", fold, "model", model_n, "logloss: ", log_loss(y_oos, pred_oos, labels=labels))
        fold += 1

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dense(10000, init='glorot_uniform'))
        model.add(PReLU())
        model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(3000, input_dim=train_nn.shape[1], init='he_uniform'))
        model.add(Activation('tanh'))
        model.add(Dense(21, init = 'he_uniform', activation='softmax'))
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(10000, input_dim=train_nn.shape[1], init='he_normal'))
        model.add(Activation('tanh'))
        model.add(Dense(21, init = 'he_normal', activation='softmax'))
        return model 


model_cv(train_nn.values, y_train, 1)

#average columns folds for test set   
test_stack.ix[:, 1:] /= nfolds
test_stack_dist.ix[:, 1:] /= nfolds

train_stack_all = pd.merge(train_stack_dist, train_stack, on='claim_id', how='inner', sort=False)
test_stack_all = pd.merge(test_stack_dist, test_stack, on='claim_id', how='inner', sort=False)

train_stack_all.to_csv('train_knn_stack.csv.gz', index = False, compression='gzip')
test_stack_all.to_csv('test_knn_stack.csv.gz', index = False, compression='gzip')

df.to_csv('keras_cv_full.csv', index = False)
