import numpy as np
np.random.seed(888)

import pandas as pd
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier

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

train_norm = pd.merge(pd.DataFrame(train[select_vars]), df_normal, on='claim_id', how='inner', sort=False)
test_norm = pd.merge(pd.DataFrame(test[select_vars]), df_normal, on='claim_id', how='inner', sort=False)

#####Sample Data#####
train_knn = train_norm.sample(frac=0.1)
y_sample = pd.merge(train[['claim_id', 'target']], train_knn[['claim_id']], on='claim_id', how='inner', sort=False)
y_train = y_sample['target'].ravel()
#####################

#y_train = train['target'].ravel()
X = train_knn.drop(['claim_id'], axis=1)
X_test = test.drop(['claim_id'], axis=1)

train_id = train_knn['claim_id'].values
test_id = test['claim_id'].values

knn_1a = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='uniform')
knn_1b = KNeighborsClassifier(n_neighbors=5, n_jobs=-1, weights='distance')

knn_1a.fit(X, y_train)
y1 = knn_1a.predict_proba(X)



knn_2 = KNeighborsClassifier(n_neighbors=10, n_jobs=-1)
knn_3 = KNeighborsClassifier(n_neighbors=25, n_jobs=-1)
knn_4 = KNeighborsClassifier(n_neighbors=50, n_jobs=-1)
knn_5 = KNeighborsClassifier(n_neighbors=100, n_jobs=-1)
knn_6 = KNeighborsClassifier(n_neighbors=250, n_jobs=-1)

knn_2c = RadiusNeighborsClassifier(radius=2, n_jobs=-1, weights='uniform')
knn_2d = RadiusNeighborsClassifier(radius=2, n_jobs=-1, weights='distance')

knn_2c.fit(X, y_train)
y2 = knn_2c.predict(X)
d2 = knn_2c.radius_neighbors(X, radius=2, return_distance=True)
d4 = knn_2c.radius_neighbors(X, radius=4, return_distance=True)

knn_2d.fit(X, y_train)
y3 = knn_2d.predict_proba(X)
