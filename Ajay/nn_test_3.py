import numpy as np
np.random.seed(888)

import feather
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
from scipy.sparse import csr_matrix
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

import tensorflow as tf
from tensorflow.python.client import device_lib
print(tf.__version__)

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
get_available_gpus()

#https://github.com/fchollet/keras/issues/3857
tf.python.control_flow_ops = tf

train = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_8.csv.gz', compression="gzip")
train.shape
test.shape

y_train = train['target'].ravel()
all_data = pd.concat([train.drop(['loss', 'target'], axis=1), test], ignore_index=True)
all_data.index
all_data.head()

#To one-hot encode
cat_cols = ["adj", "clmnt_gender", "major_class_cd", "prexist_dsblty_in", "catas_or_jntcvg_cd",
            "suit_matter_type", "initl_trtmt_cd", "state",
            "occ_code", "sic_cd", "diagnosis_icd9_cd", 
            "cas_aia_cds_1_2", "cas_aia_cds_3_4", "clm_aia_cds_1_2", "clm_aia_cds_3_4"]

#Already normalized
num_cols = ['econ_unemployment_py', 'paid_year', 'econ_gdp_py',
       'econ_gdp_ar_py', 'econ_price_py', 'econ_price_allitems_py',
       'econ_price_healthcare_py', 'econ_10yr_note_py', 
       'paid_m', 'paid_l', 'paid_o', 'any_i', 'any_m', 'any_l', 'any_o',
       'mean_i', 'mean_m', 'mean_l', 'mean_o', 'mean_im', 'mean_all', 'min_i',
       'min_m', 'min_l', 'min_o', 'min_im', 'min_all', 'max_i', 'max_m',
       'max_l', 'max_o', 'max_im', 'max_all', 'med_i', 'med_m', 'med_l',
       'med_o', 'med_im', 'med_all',
       'cutoff_year', 'avg_wkly_wage', 'imparmt_pct', 'dnb_emptotl',
       'dnb_emphere', 'dnb_worth', 'dnb_sales', 'econ_gdp_ly',
       'econ_gdp_ar_ly', 'econ_price_ly', 'econ_price_allitems_ly',
       'econ_price_healthcare_ly', 'econ_10yr_note_ly', 'econ_unemployment_ly',
       'surgery_yearmo', 'imparmt_pct_yearmo', 'clmnt_birth_yearmo',
       'empl_hire_yearmo', 'death_yearmo', 'death2_yearmo', 'loss_yearmo',
       'reported_yearmo', 'abstract_yearmo', 'clm_create_yearmo', 'mmi_yearmo',
       'rtrn_to_wrk_yearmo', 'eff_yearmo', 'exp_yearmo', 'suit_yearmo',
       'duration', 'report_to_claim', 'report_lag', 'eff_to_loss',
       'loss_to_mmi', 'loss_to_return', 'years_working', 'age_at_injury',
       'age_at_hire', 'age_at_mmi', 'age_at_return']

#To normalize
num_to_norm = ['Other', 'PPD', 'PTD', 'STLMT', 'TTD', 'n_hist']

all_data[num_cols].head()

#To normalize
mc_vars = [f for f in all_data.columns if 'mc_code' in f]
#As is
bin_vars = [f for f in all_data.columns if 'any' in f]
#To one-hot encode
cat_vars_new = [f for f in all_data.columns if 'cat' in f]

#cat_cols.extend(cat_vars_new)

df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in cat_cols:
    all_data.ix[all_data[f] > 2500, f] = 2500
    d = pd.get_dummies(all_data[f])
    frames = [df_normal, d]
    df_normal = pd.concat(frames, axis=1)

len(all_data.columns)
len(cat_cols)
len(df_normal.columns)

num_to_norm.extend(mc_vars)

#df_normal = pd.DataFrame(all_data, columns = ['claim_id'])
for f in num_to_norm:
    s = (all_data[f] - all_data[f].mean()) / all_data[f].std()
    frames = [df_normal, s]
    df_normal = pd.concat(frames, axis=1)

select_vars = num_cols
num_cols.insert(0,"claim_id")
select_vars.extend(bin_vars)

len(df_normal.columns)
df_normal.head()

train_nn = pd.merge(pd.DataFrame(train[select_vars]), df_normal, on='claim_id', how='inner', sort=False)
test_nn = pd.merge(pd.DataFrame(test[select_vars]), df_normal, on='claim_id', how='inner', sort=False)

y_train = train['target'].ravel()
train_nn.drop(['claim_id'], axis=1, inplace=True)
test_nn.drop(['claim_id'], axis=1, inplace=True)

all_data.shape
train_nn.shape
test_nn.shape
len(y_train)
to_categorical(y_train).shape

def batch_generator(X, y, batch_size, shuffle):
    #chenglong code for fiting from generator (https://www.kaggle.com/c/talkingdata-mobile-user-demographics/forums/t/22567/neural-network-for-sparse-matrices)
    number_of_batches = np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    if shuffle:
        np.random.shuffle(sample_index)
    while True:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]
        X_batch = X[batch_index,:].toarray()
        y_batch = y[batch_index]
        counter += 1
        yield X_batch, y_batch
        if (counter == number_of_batches):
            if shuffle:
                np.random.shuffle(sample_index)
            counter = 0

def batch_generatorp(X, batch_size, shuffle):
    number_of_batches = X.shape[0] / np.ceil(X.shape[0]/batch_size)
    counter = 0
    sample_index = np.arange(X.shape[0])
    while True:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]
        X_batch = X[batch_index, :].toarray()
        counter += 1
        yield X_batch
        if (counter == number_of_batches):
            counter = 0

########################################################################################################################################################


fit = model.fit_generator(generator = batch_generator(train_nn.values,to_categorical(y_train), 128, True),
                          nb_epoch = 10,
                          samples_per_epoch = train_nn.shape[0],
                          verbose = 1)


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
          nb_epoch=5, shuffle=True, validation_split=.25)
          
model = Sequential()
model.add(Dense(10000, input_dim=train_nn.shape[1], init='glorot_uniform'))
model.add(PReLU()) 
model.add(Dropout(0.1))
model.add(Dense(10000, init='glorot_uniform'))
model.add(PReLU())
model.add(Dropout(0.1))
model.add(Dense(21, init = 'glorot_uniform', activation='softmax'))
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['categorical_crossentropy'])
model.fit(train_nn.values, to_categorical(y_train), batch_size=25000, 
          nb_epoch=5, shuffle=True, validation_split=.25)
