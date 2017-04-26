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

y_train = train['loss'].ravel()
train_nn.drop(['claim_id'], axis=1, inplace=True)
test_nn.drop(['claim_id'], axis=1, inplace=True)

all_data.shape
train_nn.shape
test_nn.shape
len(y_train)

def create_model(m):
    if m==1:
        model = Sequential()
        model.add(Dense(10000, input_dim = train_nn.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.4))
        model.add(Dense(10000, init = 'he_normal'))
        model.add(PReLU())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==2:
        model = Sequential()
        model.add(Dense(400, input_dim = train_nn.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model
    if m==3:
        model = Sequential()
        model.add(Dense(400, input_dim = train_nn.shape[1], init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))
        model.add(Dense(200, init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(500, init = 'he_normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.2))
        model.add(Dense(1, init = 'he_normal'))
        model.compile(optimizer='adadelta', loss='mae')
        return model

nfolds = 2
nepochs = 5

kf = KFold(nfolds, shuffle=True)
def model_cv(X, y, model_n):
    j = 1
    pred_final = np.zeros(test_nn.shape[0])
    for train_index, test_index in kf.split(X):
        X_model, X_oos = X[train_index], X[test_index]
        y_model, y_oos = y[train_index], y[test_index]
        pred_oos = np.zeros(y_oos.shape[0])
        nn_model = create_model(model_n)
        nn_model.fit(X_model, y_model, batch_size=10000, nb_epoch=nepochs, 
                     validation_data = (X_oos, y_oos), shuffle=True)
        pred_final += nn_model.predict(test_nn.values, batch_size=10000)[:,0]
        pred_oos = nn_model.predict(X_oos)[:,0]
        print("Fold", j, "mae oos: ", np.mean(abs(pred_oos-y_oos)))
        j += 1
    pred_final /= nfolds
    return pred_final


pred_1 = model_cv(train_nn.values, y_train, 1)