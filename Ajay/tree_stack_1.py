import numpy as np
np.random.seed(888)

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor

train = pd.read_csv('train_recode_8.csv.gz', compression="gzip")
test = pd.read_csv('test_recode_8.csv.gz', compression="gzip")
train.shape
test.shape

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

train_norm = pd.merge(pd.DataFrame(train[select_vars]), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test[select_vars]), df_normal, on='claim_id', how='inner', sort=False)

#####Sample Data#####
#train_knn = train_norm.sample(frac=0.1)
#y_sample = pd.merge(train[['claim_id', 'target']], train_knn[['claim_id']], on='claim_id', how='inner', sort=False)
#y_train = y_sample['target'].ravel()
#####################

y_train = train['loss'].ravel()
X = train_norm.drop(['claim_id'], axis=1)
X_test = test_norm.drop(['claim_id'], axis=1)

train_id = train_norm['claim_id'].values
test_id = test['claim_id'].values

nfolds = 5
kf = KFold(nfolds, shuffle=True)

boost_1 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2, splitter='best'), n_estimators = 250)
boost_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=2, splitter='random'), n_estimators = 250)

etr_1 = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, max_features='sqrt')
etr_2 = ExtraTreesRegressor(n_estimators=500, n_jobs=-1, max_features='auto')
etr_3 = ExtraTreesRegressor(n_estimators=100, max_depth=5, n_jobs=-1, max_features='sqrt')
etr_4 = ExtraTreesRegressor(n_estimators=100, max_depth=5, n_jobs=-1, max_features='auto')

boost_3 = AdaBoostRegressor(etr_3, n_estimators = 250, learning_rate=.5)
boost_4 = AdaBoostRegressor(etr_4, n_estimators = 250, learning_rate=.5)

rf_1 = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_features='sqrt')
rf_2 = RandomForestRegressor(n_estimators=500, n_jobs=-1, max_features='auto')

models = [rf_1, rf_2, etr_1, etr_2, boost_1, boost_2, boost_3, boost_4]
#models = [rf_1, etr_1]

#To store stacked values
train_stack = pd.DataFrame(train_id, columns = ['claim_id'])
test_stack = pd.DataFrame(test_id, columns = ['claim_id'])

blend_cols_tr = pd.DataFrame(np.zeros(shape=(train_stack.shape[0], len(models))))
blend_cols_te = pd.DataFrame(np.zeros(shape=(test_stack.shape[0], len(models))))
train_stack = pd.concat([train_stack, blend_cols_tr], axis=1)
test_stack = pd.concat([test_stack, blend_cols_te], axis=1)

fold = 1
for train_index, test_index in kf.split(X):
    X_model, X_oos = X.values[train_index], X.values[test_index]
    y_model, y_oos = y_train[train_index], y_train[test_index]
    mod = 1
    for i in models:
        i.fit(X_model, y_model)
        train_stack.ix[test_index, mod] = i.predict(X_oos)
        test_stack.ix[:, mod] += i.predict(X_test.values)
        pred_oos = i.predict(X_oos)
        print("Fold", fold, "model", mod, "MSE: ", mean_squared_error(y_oos, pred_oos))
        mod += 1
    fold += 1

#average columns folds for test set
test_stack.ix[:, 1:] /= nfolds

train_stack.to_csv('train_stack_loss1.csv.gz', index = False, compression='gzip')
test_stack.to_csv('test_stack_loss1.csv.gz', index = False, compression='gzip')

