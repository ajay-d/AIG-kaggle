'''
keras_12
'''
from __future__ import division
version = 'keras_12_hold0.234477_all'

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
import subprocess
from scipy.sparse import csr_matrix, hstack
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import KFold, train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping

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
            
def cnt(dsn, varlist):
    df = dsn[varlist]
    df['cnt'] = 1.0
    return df.groupby(varlist).transform(np.sum).cnt
        

def loo(dsn, varlist, y, split, r_k=.3):
    
    #median of target as complement
    y0 = np.mean(y[split])
    
    # table for all records
    df1 = dsn[varlist]
    df1['y'] = y
    
    # table for training only
    df2 = df1[split]
    df2['cnt'] = 1.0
    
    # calculate grouped total y and count
    grouped = df2.groupby(varlist).sum().add_prefix('sum_')
    
    # merge back to whole table
    df1 = pd.merge(df1, grouped, left_on = varlist, right_index = True, how = 'left')
    
    # leave one out
    df1['sum_cnt'] = np.where(split, df1['sum_cnt']-1, df1['sum_cnt'])
    df1['sum_y'] = np.where(split, df1['sum_y']-df1['y'], df1['sum_y'])
    
    # calculated transformed cats based on credibilty
    cred_k = 2
    
    df1['loo_y'] = (df1['sum_y'] + y0*cred_k)*1.0 / (df1['sum_cnt'] + cred_k)
    df1['loo_y'][df1['loo_y'].isnull()] = y0
    
    # add noise to pretent overfitting
    df1['loo_y'][split] = df1['loo_y'][split]*(1+(np.random.uniform(0,1,sum(split))-0.5)*r_k)
    return df1['loo_y']

claim = pd.read_csv('/mnt/aigdata/OriginalData/claims.csv', encoding='latin1')
train = pd.read_csv('/mnt/aigdata/OriginalData/train.csv', encoding='latin1')
test = pd.read_csv('/mnt/aigdata/OriginalData/test.csv', encoding='latin1')


test_id = test[['claim_id']]
test_id['split'] = -1

# covert years

claim['age_at_injury'] = claim.loss_yearmo - np.where((claim['clmnt_birth_yearmo']<=2015)&(claim['clmnt_birth_yearmo']>1900), claim['clmnt_birth_yearmo'], np.nan)
claim['age_at_injury'][claim['age_at_injury']<0] = np.nan 

claim['empl_tenure_at_injury'] = claim.loss_yearmo - np.where(((claim['empl_hire_yearmo']<=2015)&(claim['empl_hire_yearmo']>1900)), claim['empl_hire_yearmo'], np.nan)
claim['empl_tenure_at_injury'][claim['empl_tenure_at_injury']<0] = 0

claim['report_lag'] = np.maximum(0, claim.reported_yearmo - claim.loss_yearmo) 

claim.initl_trtmt_cd = claim.initl_trtmt_cd.astype(str)

claim.sic_cd = claim.sic_cd.str.replace('"', '')
claim.sic_cd = claim.sic_cd.str.replace("'", '')
claim.sic_cd = claim.sic_cd.str.replace('*', '')
claim.sic_cd = claim.sic_cd.str.replace('-', '')
claim.sic_cd = claim.sic_cd.str.replace('[A-Z]', '')
claim.sic_cd = claim.sic_cd.str.replace('.', '')

# interactions
claim.state[claim.state.isnull()] = '_M'

#claim['state_loss_yr'] = claim.state + np.modf(claim['loss_yearmo'])[1].astype(str)

claim['wage_grp'] = pd.qcut(claim.avg_wkly_wage, 10)
claim['wage_grp'] = claim['wage_grp'].astype(str)
claim['wage_grp'][claim['wage_grp'] == 'nan'] = '_M'

claim['state_wage'] = claim.state + claim.wage_grp

claim.clm_aia_cds_1_2[claim.clm_aia_cds_1_2.isnull()] = '_M'
claim.clm_aia_cds_3_4[claim.clm_aia_cds_3_4.isnull()] = '_M'
claim.cas_aia_cds_1_2[claim.cas_aia_cds_1_2.isnull()] = '_M'
claim.cas_aia_cds_3_4[claim.cas_aia_cds_3_4.isnull()] = '_M'

claim['bpic'] = claim.clm_aia_cds_1_2 + claim.clm_aia_cds_3_4
claim['cause_detri_code'] = claim.cas_aia_cds_1_2 + claim.cas_aia_cds_3_4
claim.occ_code = claim.occ_code.astype(str)
claim['occ_ic'] = claim.occ_code + claim.clm_aia_cds_3_4
claim['occ_bp'] = claim.occ_code + claim.clm_aia_cds_1_2
claim['bp_cause_code'] = claim.clm_aia_cds_1_2 + claim.cas_aia_cds_1_2
claim['ic_cause_code'] = claim.clm_aia_cds_3_4 + claim.cas_aia_cds_1_2
claim['bp_detri_code'] = claim.clm_aia_cds_1_2 + claim.cas_aia_cds_3_4
claim['ic_detri_code'] = claim.clm_aia_cds_3_4 + claim.cas_aia_cds_3_4

claim['state_bp'] = claim.state + claim.clm_aia_cds_1_2
claim['state_ic'] = claim.state + claim.clm_aia_cds_3_4
claim['state_cause'] = claim.state + claim.cas_aia_cds_1_2
claim['state_detri'] = claim.state + claim.cas_aia_cds_3_4

claim.diagnosis_icd9_cd[claim.diagnosis_icd9_cd.isnull()] = '____M'
claim['icd9_d1'] = [x[:1] for x in claim.diagnosis_icd9_cd]
claim.icd9_d1[claim.icd9_d1.isnull()] = '_'
claim['icd9_d12'] = [x[:2] for x in claim.diagnosis_icd9_cd]
claim['icd9_d123'] = [x[:3] for x in claim.diagnosis_icd9_cd]


claim['state_icd9'] = claim.state + claim.diagnosis_icd9_cd
claim['bp_icd9'] = claim.clm_aia_cds_1_2 + claim.diagnosis_icd9_cd
claim['ic_icd9'] = claim.clm_aia_cds_3_4 + claim.diagnosis_icd9_cd
claim['cause_icd9'] = claim.cas_aia_cds_1_2 + claim.diagnosis_icd9_cd
claim['detri_icd9'] = claim.cas_aia_cds_3_4 + claim.diagnosis_icd9_cd


var_claim = ['claim_id', 'slice', 'clmnt_gender','avg_wkly_wage',\
             'imparmt_pct', 'initl_trtmt_cd','prexist_dsblty_in','catas_or_jntcvg_cd',\
             'age_at_injury','empl_tenure_at_injury','state','suit_matter_type']
var_year = ['surgery_yearmo','imparmt_pct_yearmo','loss_yearmo','mmi_yearmo',\
            'rtrn_to_wrk_yearmo','suit_yearmo','death2_yearmo', 'reported_yearmo', 'clm_create_yearmo']
var_lag = ['report_lag']
var_code = ['diagnosis_icd9_cd','icd9_d1','icd9_d12','icd9_d123','cas_aia_cds_1_2','cas_aia_cds_3_4','clm_aia_cds_1_2','clm_aia_cds_3_4',\
            'sic_cd','occ_code']
var_interaction = ['bpic','bp_cause_code','ic_cause_code','cause_detri_code', 'bp_detri_code', 'ic_detri_code',\
                   'state_wage','occ_ic','occ_bp','state_bp','state_ic','state_cause','state_detri',\
                  'state_icd9', 'bp_icd9','ic_icd9','cause_icd9','detri_icd9']
var_law = ['law_limit_tt','law_limit_pt','law_limit_pp','law_cola','law_offsets','law_ib_scheduled']
    
var_list = var_claim + var_year + var_lag + var_code + var_interaction + var_law


############################
## Merge train test claim ##
############################

train_test = train.append(test)
train_test = pd.merge(train_test, claim[var_list], how='inner', on=['claim_id'])
del(train, test, claim)

train_test.slice = np.where(train_test.slice>10, train_test.slice-10, train_test.slice)

print('>>> creating years and lags')

train_test['cutoff_year'] = train_test.paid_year - train_test.slice

train_test.surgery_yearmo = np.where(train_test.cutoff_year < train_test.surgery_yearmo, np.nan, train_test.surgery_yearmo)

train_test.imparmt_pct_yearmo = np.where(train_test.cutoff_year < train_test.imparmt_pct_yearmo, np.nan, train_test.imparmt_pct_yearmo)
train_test.imparmt_pct = np.where(train_test.imparmt_pct_yearmo.isnull(), np.nan, train_test.imparmt_pct)

train_test.death2_yearmo = np.where(train_test.cutoff_year < train_test.death2_yearmo, np.nan, train_test.death2_yearmo)
#train_test.death_ind = np.where(train_test.death2_yearmo.isnull(), 0, 1)

train_test.mmi_yearmo = np.where(train_test.cutoff_year < train_test.mmi_yearmo, np.nan, train_test.mmi_yearmo)

train_test.rtrn_to_wrk_yearmo = np.where(train_test.cutoff_year < train_test.rtrn_to_wrk_yearmo, np.nan, train_test.rtrn_to_wrk_yearmo)

train_test.suit_yearmo = np.where(train_test.cutoff_year < train_test.suit_yearmo, np.nan, train_test.suit_yearmo)
train_test.suit_matter_type[train_test.suit_matter_type.str.strip()=='.'] = '_M'
train_test.suit_matter_type.value_counts()
train_test.suit_matter_type = np.where(train_test.suit_yearmo.isnull(), '_M', train_test.suit_matter_type)

train_test['mmi_report_lag'] = np.where(train_test.mmi_yearmo.isnull(), np.nan, np.maximum(0, train_test.mmi_yearmo - train_test.reported_yearmo)) 
train_test['surgury_report_lag'] = np.where(train_test.surgery_yearmo.isnull(), np.nan, np.maximum(0, train_test.surgery_yearmo - train_test.reported_yearmo)) 
train_test['imparmt_report_lag'] = np.where(train_test.imparmt_pct_yearmo.isnull(), np.nan, np.maximum(0, train_test.imparmt_pct_yearmo - train_test.reported_yearmo)) 

train_test['mmi_cutoff_lag'] = np.where(train_test.mmi_yearmo.isnull(), np.nan, (2015 - train_test.slice) - train_test.mmi_yearmo) 
train_test['surgury_cutoff_lag'] = np.where(train_test.surgery_yearmo.isnull(), np.nan, (2015 - train_test.slice) - train_test.surgery_yearmo) 
train_test['imparmt_cutoff_lag'] = np.where(train_test.imparmt_pct_yearmo.isnull(), np.nan, (2015 - train_test.slice) - train_test.imparmt_pct_yearmo) 

train_test['suit_ind'] = np.where(train_test['suit_yearmo'].isnull(), 0, 1)


train_test['state_suit_type'] = train_test.state + train_test.suit_matter_type
train_test['suit_type_yr'] = train_test.suit_matter_type + np.where(train_test.suit_yearmo.isnull(), '_M', np.modf(train_test.suit_yearmo)[1].astype(str))

train_test_2015 = train_test[train_test.paid_year == 2015].reset_index(drop=True)

print('>>> first payment year')
first_paid = train_test[['claim_id','paid_year']][(2015 - train_test.slice) >= train_test.paid_year]
first_paid.sort_values(by=['claim_id','paid_year'], axis=0, inplace=True)
first_paid.drop_duplicates(['claim_id'], inplace=True, keep='first')
first_paid.columns = ['claim_id', 'first_paid_year']
train_test_2015 = pd.merge(train_test_2015, first_paid, on=['claim_id'], how='left')

del(first_paid)

print('>>> first non zero payment year')

first_non0_paid = train_test[['claim_id','paid_year','paid_m','paid_i']][(2015 - train_test.slice) >= train_test.paid_year]
first_non0_paid['paid_mi'] = first_non0_paid.paid_m + first_non0_paid.paid_i
first_non0_paid['first_non0_paid_yr'] = first_non0_paid.paid_year 

first_non0_paid = first_non0_paid[first_non0_paid['paid_mi']>0]
first_non0_paid_yr = first_non0_paid[['claim_id', 'first_non0_paid_yr']]
first_non0_paid_yr.sort_values(by=['claim_id','first_non0_paid_yr'], axis=0, inplace=True)
first_non0_paid_yr.drop_duplicates(['claim_id'], inplace=True, keep='first')
train_test_2015 = pd.merge(train_test_2015, first_non0_paid_yr, on=['claim_id'], how='left')
                         
del(first_non0_paid, first_non0_paid_yr)   


print('>>> last non zero payment year')

last_non0_paid = train_test[['claim_id','paid_year','paid_m','paid_i']][(2015 - train_test.slice) >= train_test.paid_year]
last_non0_paid['paid_mi'] = last_non0_paid.paid_m + last_non0_paid.paid_i
last_non0_paid['last_non0_paid_yr'] = last_non0_paid.paid_year 

last_non0_paid = last_non0_paid[last_non0_paid['paid_mi']>0]
last_non0_paid_yr = last_non0_paid[['claim_id', 'last_non0_paid_yr']]
last_non0_paid_yr.sort_values(by=['claim_id','last_non0_paid_yr'], axis=0, inplace=True)
last_non0_paid_yr.drop_duplicates(['claim_id'], inplace=True, keep='last')
train_test = pd.merge(train_test, last_non0_paid_yr, on=['claim_id'], how='left')
train_test_2015 = pd.merge(train_test_2015, last_non0_paid_yr, on=['claim_id'], how='left')
                         
del(last_non0_paid, last_non0_paid_yr)    

train_test_2015['mmi_last_non0_lag'] = np.where(train_test_2015.mmi_yearmo.isnull(), np.nan, train_test_2015.last_non0_paid_yr - train_test_2015.mmi_yearmo) 
train_test_2015['surgury_last_non0_lag'] = np.where(train_test_2015.surgery_yearmo.isnull(), np.nan, train_test_2015.last_non0_paid_yr - train_test_2015.surgery_yearmo) 
train_test_2015['imparmt_last_non0_lag'] = np.where(train_test_2015.imparmt_pct_yearmo.isnull(), np.nan, train_test_2015.last_non0_paid_yr - train_test_2015.imparmt_pct_yearmo) 


print('>>> econ variables from last_non0_paid_yr date to 2015')

econ_vars = ['econ_gdp_py','econ_price_py','econ_price_allitems_py','econ_price_healthcare_py']
              
econ_vars_paid_year = train_test[['paid_year']+econ_vars][train_test.paid_year>=1981]
econ_vars_paid_year.drop_duplicates(['paid_year']+econ_vars, inplace=True)
econ_vars_paid_year.sort_values(by=['paid_year'], ascending=False, inplace=True)
econ_vars_paid_year.econ_gdp_py = econ_vars_paid_year.econ_gdp_py.cumsum()
econ_vars_paid_year.econ_price_py = econ_vars_paid_year.econ_price_py.cumsum()
econ_vars_paid_year.econ_price_allitems_py = econ_vars_paid_year.econ_price_allitems_py.cumsum()
econ_vars_paid_year.econ_price_healthcare_py = econ_vars_paid_year.econ_price_healthcare_py.cumsum()
econ_vars_paid_year = econ_vars_paid_year.add_suffix('_cumsum')

train_test_2015 = pd.merge(train_test_2015, econ_vars_paid_year, left_on='last_non0_paid_yr', right_on='paid_year_cumsum', how='left')
train_test_2015.drop(['paid_year_cumsum'], axis=1, inplace=True)

del(econ_vars, econ_vars_paid_year)

train_test_2015['duration_2_last_non0_paid_yr'] = np.where(train_test_2015.last_non0_paid_yr.isnull(), np.nan, np.maximum(train_test_2015.last_non0_paid_yr - train_test_2015.loss_yearmo, 0))
train_test_2015['duration_2_cutoff'] = np.where(train_test_2015.cutoff_year.isnull(), np.nan, np.maximum(train_test_2015.cutoff_year - train_test_2015.loss_yearmo, 0))


print('>>> create paid amount to date')

mi_paid = train_test[['claim_id', 'paid_m', 'paid_i', 'paid_l', 'paid_o']][(2015 - train_test.slice) >= train_test.paid_year]
mi_paid['paid_mi'] =  mi_paid.paid_m + mi_paid.paid_i
mi_paid['paid_milo'] =  mi_paid.paid_m + mi_paid.paid_i + mi_paid.paid_l + mi_paid.paid_o
mi_paid.drop(['paid_o'], axis=1, inplace=True)

mi_paid_2_date_max = mi_paid.groupby('claim_id').max().add_suffix('_2_date_max')
train_test_2015 = pd.merge(train_test_2015, mi_paid_2_date_max, left_on='claim_id', right_index=True, how='left')
del(mi_paid_2_date_max)

mi_paid_2_date_min = mi_paid.groupby('claim_id').min().add_suffix('_2_date_min')
train_test_2015 = pd.merge(train_test_2015, mi_paid_2_date_min, left_on='claim_id', right_index=True, how='left')
del(mi_paid_2_date_min)

mi_paid_2_date_mean = mi_paid.groupby('claim_id').mean().add_suffix('_2_date_mean')
train_test_2015 = pd.merge(train_test_2015, mi_paid_2_date_mean, left_on='claim_id', right_index=True, how='left')
del(mi_paid_2_date_mean)

mi_paid = train_test[['claim_id', 'paid_m', 'paid_i', 'paid_l', 'paid_o','adj']][(2015 - train_test.slice) >= train_test.paid_year]
mi_paid['paid_mi'] =  mi_paid.paid_m + mi_paid.paid_i
mi_paid['paid_milo'] =  mi_paid.paid_m + mi_paid.paid_i + mi_paid.paid_l + mi_paid.paid_o
mi_paid.drop(['paid_o'], axis=1, inplace=True)

mi_paid['paid_m_non0'] = np.where(mi_paid.paid_m>0, 1, 0)
mi_paid['paid_i_non0'] = np.where(mi_paid.paid_i>0, 1, 0)
mi_paid['paid_l_non0'] = np.where(mi_paid.paid_l>0, 1, 0)
mi_paid['paid_mi_non0'] = np.where(mi_paid.paid_mi>0, 1, 0)
mi_paid['paid_milo_non0'] = np.where(mi_paid.paid_milo>0, 1, 0)

for adju in ['Other','PPD','PTD','STLMT','TTD']:
    mi_paid['adj_'+adju] = np.where(mi_paid.adj.str.contains(adju),1,0)
    
mi_paid.drop(['adj'],axis=1,inplace=True)

mi_paid_2_date = mi_paid.groupby('claim_id').sum().add_suffix('_2_date')
train_test_2015 = pd.merge(train_test_2015, mi_paid_2_date, left_on='claim_id', right_index=True, how='left')

del(mi_paid, mi_paid_2_date)

train_test_2015.paid_m_2_date = np.maximum(train_test_2015.paid_m_2_date, 0)
train_test_2015.paid_i_2_date = np.maximum(train_test_2015.paid_i_2_date, 0)
train_test_2015.paid_mi_2_date = np.maximum(train_test_2015.paid_mi_2_date, 0)
train_test_2015.paid_l_2_date = np.maximum(train_test_2015.paid_l_2_date, 0)
train_test_2015.paid_milo_2_date = np.maximum(train_test_2015.paid_milo_2_date, 0)

train_test_2015['paid_m_ratio'] = np.where(train_test_2015['paid_mi_2_date']>0, train_test_2015.paid_m_2_date/train_test_2015.paid_mi_2_date, 0)
train_test_2015['paid_mi_ratio'] = np.where(train_test_2015['paid_milo_2_date']>0, train_test_2015.paid_mi_2_date/train_test_2015.paid_milo_2_date, 0)

train_test_2015.paid_m_ratio = np.log(train_test_2015.paid_m_ratio + 1)
train_test_2015.paid_mi_ratio = np.log(train_test_2015.paid_mi_ratio + 1)

train_test_2015['paid_m_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_m_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_i_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_i_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_l_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_l_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_mi_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_mi_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_milo_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_milo_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)

print('>>> prior 10 medical and indemnity payment')
prior_10 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+10))]
prior_10['paid_mi'] = prior_10.paid_m + prior_10.paid_i
prior_10['paid_milo'] = prior_10.paid_m + prior_10.paid_i + prior_10.paid_l + prior_10.paid_o
prior_10.drop(['paid_o'], axis=1, inplace=True)

prior_10_sum = prior_10.groupby('claim_id').sum().add_prefix('prior_10_sum_')
train_test_2015 = pd.merge(train_test_2015, prior_10_sum, left_on='claim_id', right_index=True, how='left')
del(prior_10_sum)

prior_10_max = prior_10.groupby('claim_id').max().add_prefix('prior_10_max_')
train_test_2015 = pd.merge(train_test_2015, prior_10_max, left_on='claim_id', right_index=True, how='left')
del(prior_10_max)

prior_10_min = prior_10.groupby('claim_id').min().add_prefix('prior_10_min_')
train_test_2015 = pd.merge(train_test_2015, prior_10_min, left_on='claim_id', right_index=True, how='left')
del(prior_10_min)

prior_10_mean = prior_10.groupby('claim_id').mean().add_prefix('prior_10_mean_')
train_test_2015 = pd.merge(train_test_2015, prior_10_mean, left_on='claim_id', right_index=True, how='left')
del(prior_10_mean)

del(prior_10)

train_test_2015.prior_10_sum_paid_mi = np.log(np.maximum(train_test_2015.prior_10_sum_paid_mi, 0) + 1)
train_test_2015.prior_10_sum_paid_milo = np.log(np.maximum(train_test_2015.prior_10_sum_paid_milo, 0) + 1)
train_test_2015.prior_10_sum_paid_m = np.log(np.maximum(train_test_2015.prior_10_sum_paid_m, 0) + 1)
train_test_2015.prior_10_sum_paid_i = np.log(np.maximum(train_test_2015.prior_10_sum_paid_i, 0) + 1)
train_test_2015.prior_10_sum_paid_l = np.log(np.maximum(train_test_2015.prior_10_sum_paid_l, 0) + 1)

train_test_2015.prior_10_max_paid_mi = np.log(np.maximum(train_test_2015.prior_10_max_paid_mi, 0) + 1)
train_test_2015.prior_10_max_paid_milo = np.log(np.maximum(train_test_2015.prior_10_max_paid_milo, 0) + 1)
train_test_2015.prior_10_max_paid_m = np.log(np.maximum(train_test_2015.prior_10_max_paid_m, 0) + 1)
train_test_2015.prior_10_max_paid_i = np.log(np.maximum(train_test_2015.prior_10_max_paid_i, 0) + 1)
train_test_2015.prior_10_max_paid_l = np.log(np.maximum(train_test_2015.prior_10_max_paid_l, 0) + 1)

train_test_2015.prior_10_min_paid_mi = np.log(np.maximum(train_test_2015.prior_10_min_paid_mi, 0) + 1)
train_test_2015.prior_10_min_paid_milo = np.log(np.maximum(train_test_2015.prior_10_min_paid_milo, 0) + 1)
train_test_2015.prior_10_min_paid_m = np.log(np.maximum(train_test_2015.prior_10_min_paid_m, 0) + 1)
train_test_2015.prior_10_min_paid_i = np.log(np.maximum(train_test_2015.prior_10_min_paid_i, 0) + 1)
train_test_2015.prior_10_min_paid_l = np.log(np.maximum(train_test_2015.prior_10_min_paid_l, 0) + 1)
 
train_test_2015.prior_10_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_10_mean_paid_mi, 0) + 1)
train_test_2015.prior_10_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_10_mean_paid_milo, 0) + 1)
train_test_2015.prior_10_mean_paid_m = np.log(np.maximum(train_test_2015.prior_10_mean_paid_m, 0) + 1)
train_test_2015.prior_10_mean_paid_i = np.log(np.maximum(train_test_2015.prior_10_mean_paid_i, 0) + 1)
train_test_2015.prior_10_mean_paid_l = np.log(np.maximum(train_test_2015.prior_10_mean_paid_l, 0) + 1)

print('>>> prior 5 medical and indemnity payment')
prior_5 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+5))]
prior_5['paid_mi'] = prior_5.paid_m + prior_5.paid_i
prior_5['paid_milo'] = prior_5.paid_m + prior_5.paid_i + prior_5.paid_l + prior_5.paid_o
prior_5.drop(['paid_o'], axis=1, inplace=True)
   
prior_5_sum = prior_5.groupby('claim_id').sum().add_prefix('prior_5_sum_')
train_test_2015 = pd.merge(train_test_2015, prior_5_sum, left_on='claim_id', right_index=True, how='left')
del(prior_5_sum)

prior_5_max = prior_5.groupby('claim_id').max().add_prefix('prior_5_max_')
train_test_2015 = pd.merge(train_test_2015, prior_5_max, left_on='claim_id', right_index=True, how='left')
del(prior_5_max)

prior_5_min = prior_5.groupby('claim_id').min().add_prefix('prior_5_min_')
train_test_2015 = pd.merge(train_test_2015, prior_5_min, left_on='claim_id', right_index=True, how='left')
del(prior_5_min)

prior_5_mean = prior_5.groupby('claim_id').mean().add_prefix('prior_5_mean_')
train_test_2015 = pd.merge(train_test_2015, prior_5_mean, left_on='claim_id', right_index=True, how='left')
del(prior_5_mean)

del(prior_5)

train_test_2015.prior_5_sum_paid_mi = np.log(np.maximum(train_test_2015.prior_5_sum_paid_mi, 0) + 1)
train_test_2015.prior_5_sum_paid_milo = np.log(np.maximum(train_test_2015.prior_5_sum_paid_milo, 0) + 1)
train_test_2015.prior_5_sum_paid_m = np.log(np.maximum(train_test_2015.prior_5_sum_paid_m, 0) + 1)
train_test_2015.prior_5_sum_paid_i = np.log(np.maximum(train_test_2015.prior_5_sum_paid_i, 0) + 1)
train_test_2015.prior_5_sum_paid_l = np.log(np.maximum(train_test_2015.prior_5_sum_paid_l, 0) + 1)

train_test_2015.prior_5_max_paid_mi = np.log(np.maximum(train_test_2015.prior_5_max_paid_mi, 0) + 1)
train_test_2015.prior_5_max_paid_milo = np.log(np.maximum(train_test_2015.prior_5_max_paid_milo, 0) + 1)
train_test_2015.prior_5_max_paid_m = np.log(np.maximum(train_test_2015.prior_5_max_paid_m, 0) + 1)
train_test_2015.prior_5_max_paid_i = np.log(np.maximum(train_test_2015.prior_5_max_paid_i, 0) + 1)
train_test_2015.prior_5_max_paid_l = np.log(np.maximum(train_test_2015.prior_5_max_paid_l, 0) + 1)

train_test_2015.prior_5_min_paid_mi = np.log(np.maximum(train_test_2015.prior_5_min_paid_mi, 0) + 1)
train_test_2015.prior_5_min_paid_milo = np.log(np.maximum(train_test_2015.prior_5_min_paid_milo, 0) + 1)
train_test_2015.prior_5_min_paid_m = np.log(np.maximum(train_test_2015.prior_5_min_paid_m, 0) + 1)
train_test_2015.prior_5_min_paid_i = np.log(np.maximum(train_test_2015.prior_5_min_paid_i, 0) + 1)
train_test_2015.prior_5_min_paid_l = np.log(np.maximum(train_test_2015.prior_5_min_paid_l, 0) + 1)
 
train_test_2015.prior_5_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_5_mean_paid_mi, 0) + 1)
train_test_2015.prior_5_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_5_mean_paid_milo, 0) + 1)
train_test_2015.prior_5_mean_paid_m = np.log(np.maximum(train_test_2015.prior_5_mean_paid_m, 0) + 1)
train_test_2015.prior_5_mean_paid_i = np.log(np.maximum(train_test_2015.prior_5_mean_paid_i, 0) + 1)
train_test_2015.prior_5_mean_paid_l = np.log(np.maximum(train_test_2015.prior_5_mean_paid_l, 0) + 1)


print('>>> prior 3 medical and indemnity payment')
prior_3 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+3))]
prior_3['paid_mi'] = prior_3.paid_m + prior_3.paid_i
prior_3['paid_milo'] = prior_3.paid_m + prior_3.paid_i + prior_3.paid_l + prior_3.paid_o
prior_3.drop(['paid_o'], axis=1, inplace=True)
  
prior_3_sum = prior_3.groupby('claim_id').sum().add_prefix('prior_3_sum_')
train_test_2015 = pd.merge(train_test_2015, prior_3_sum, left_on='claim_id', right_index=True, how='left')
del(prior_3_sum)

prior_3_max = prior_3.groupby('claim_id').max().add_prefix('prior_3_max_')
train_test_2015 = pd.merge(train_test_2015, prior_3_max, left_on='claim_id', right_index=True, how='left')
del(prior_3_max)

prior_3_min = prior_3.groupby('claim_id').min().add_prefix('prior_3_min_')
train_test_2015 = pd.merge(train_test_2015, prior_3_min, left_on='claim_id', right_index=True, how='left')
del(prior_3_min)

prior_3_mean = prior_3.groupby('claim_id').mean().add_prefix('prior_3_mean_')
train_test_2015 = pd.merge(train_test_2015, prior_3_mean, left_on='claim_id', right_index=True, how='left')
del(prior_3_mean)

del(prior_3)

train_test_2015['prior_3_sum_paid_m_2_date'] = np.where(train_test_2015.paid_m_2_date>0, np.maximum(train_test_2015.prior_3_sum_paid_m, 0)/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['prior_3_sum_paid_i_2_date'] = np.where(train_test_2015.paid_i_2_date>0, np.maximum(train_test_2015.prior_3_sum_paid_i, 0)/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['prior_3_sum_paid_l_2_date'] = np.where(train_test_2015.paid_l_2_date>0, np.maximum(train_test_2015.prior_3_sum_paid_l, 0)/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['prior_3_sum_paid_mi_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, np.maximum(train_test_2015.prior_3_sum_paid_mi, 0)/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['prior_3_sum_paid_milo_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, np.maximum(train_test_2015.prior_3_sum_paid_milo, 0)/train_test_2015.paid_milo_2_date, np.nan)

train_test_2015.prior_3_sum_paid_mi = np.log(np.maximum(train_test_2015.prior_3_sum_paid_mi, 0) + 1)
train_test_2015.prior_3_sum_paid_milo = np.log(np.maximum(train_test_2015.prior_3_sum_paid_milo, 0) + 1)
train_test_2015.prior_3_sum_paid_m = np.log(np.maximum(train_test_2015.prior_3_sum_paid_m, 0) + 1)
train_test_2015.prior_3_sum_paid_i = np.log(np.maximum(train_test_2015.prior_3_sum_paid_i, 0) + 1)
train_test_2015.prior_3_sum_paid_l = np.log(np.maximum(train_test_2015.prior_3_sum_paid_l, 0) + 1)

train_test_2015.prior_3_max_paid_mi = np.log(np.maximum(train_test_2015.prior_3_max_paid_mi, 0) + 1)
train_test_2015.prior_3_max_paid_milo = np.log(np.maximum(train_test_2015.prior_3_max_paid_milo, 0) + 1)
train_test_2015.prior_3_max_paid_m = np.log(np.maximum(train_test_2015.prior_3_max_paid_m, 0) + 1)
train_test_2015.prior_3_max_paid_i = np.log(np.maximum(train_test_2015.prior_3_max_paid_i, 0) + 1)
train_test_2015.prior_3_max_paid_l = np.log(np.maximum(train_test_2015.prior_3_max_paid_l, 0) + 1)

train_test_2015.prior_3_min_paid_mi = np.log(np.maximum(train_test_2015.prior_3_min_paid_mi, 0) + 1)
train_test_2015.prior_3_min_paid_milo = np.log(np.maximum(train_test_2015.prior_3_min_paid_milo, 0) + 1)
train_test_2015.prior_3_min_paid_m = np.log(np.maximum(train_test_2015.prior_3_min_paid_m, 0) + 1)
train_test_2015.prior_3_min_paid_i = np.log(np.maximum(train_test_2015.prior_3_min_paid_i, 0) + 1)
train_test_2015.prior_3_min_paid_l = np.log(np.maximum(train_test_2015.prior_3_min_paid_l, 0) + 1)
 
train_test_2015.prior_3_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_3_mean_paid_mi, 0) + 1)
train_test_2015.prior_3_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_3_mean_paid_milo, 0) + 1)
train_test_2015.prior_3_mean_paid_m = np.log(np.maximum(train_test_2015.prior_3_mean_paid_m, 0) + 1)
train_test_2015.prior_3_mean_paid_i = np.log(np.maximum(train_test_2015.prior_3_mean_paid_i, 0) + 1)
train_test_2015.prior_3_mean_paid_l = np.log(np.maximum(train_test_2015.prior_3_mean_paid_l, 0) + 1)

print('>>> prior 1,2,3,4,5,6 medical and indemnity payment')

for i in [1,2,3,4,5,6]:
    prior = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) == (train_test.paid_year+i))]
    prior['paid_mi'] = prior['paid_m'] + prior['paid_i']
    prior['paid_milo'] = prior['paid_m'] + prior['paid_i'] + prior['paid_l'] + prior['paid_o']
    
    prior = prior.add_suffix('_prior_'+str(i))      
    prior['claim_id'] = prior['claim_id_prior_'+str(i)]
    prior.drop(['claim_id_prior_'+str(i), 'paid_o_prior_'+str(i)],1,inplace=True)
    if i == 1:
        prior_info = prior
    else:
        prior_info = pd.merge(prior_info, prior, on=['claim_id'], how='outer')  
        
train_test_2015 = pd.merge(train_test_2015, prior_info, on=['claim_id'], how='left')
del(prior, prior_info)

for i in [1,2,3,4,5,6]:
    train_test_2015['paid_m_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_m_2_date>0, np.maximum(train_test_2015['paid_m_prior_'+str(i)], 0)/train_test_2015.paid_m_2_date, np.nan)
    train_test_2015['paid_i_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_i_2_date>0, np.maximum(train_test_2015['paid_i_prior_'+str(i)], 0)/train_test_2015.paid_i_2_date, np.nan)
    train_test_2015['paid_l_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_l_2_date>0, np.maximum(train_test_2015['paid_l_prior_'+str(i)], 0)/train_test_2015.paid_l_2_date, np.nan)
    train_test_2015['paid_mi_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_mi_2_date>0, np.maximum(train_test_2015['paid_mi_prior_'+str(i)], 0)/train_test_2015.paid_mi_2_date, np.nan)
    train_test_2015['paid_milo_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_milo_2_date>0, np.maximum(train_test_2015['paid_milo_prior_'+str(i)], 0)/train_test_2015.paid_milo_2_date, np.nan)
        
    train_test_2015['paid_mi_prior_'+str(i)] = np.log(np.maximum(train_test_2015['paid_mi_prior_'+str(i)], 0) + 1)
    train_test_2015['paid_milo_prior_'+str(i)] = np.log(np.maximum(train_test_2015['paid_milo_prior_'+str(i)], 0) + 1)
    train_test_2015['paid_m_prior_'+str(i)] = np.log(np.maximum(train_test_2015['paid_m_prior_'+str(i)], 0) + 1)
    train_test_2015['paid_i_prior_'+str(i)] = np.log(np.maximum(train_test_2015['paid_i_prior_'+str(i)], 0) + 1)
    train_test_2015['paid_l_prior_'+str(i)] = np.log(np.maximum(train_test_2015['paid_l_prior_'+str(i)], 0) + 1)

print('>>> mc string to date')
mc_string = train_test[['claim_id', 'mc_string']][(2015 - train_test.slice) >= train_test.paid_year]
mc_string['mc_string_counts'] = np.where(mc_string['mc_string'].isnull(), 0, (mc_string['mc_string'].str.count(' ')+1))
    
mc_string_2_date = mc_string.groupby('claim_id')[['mc_string_counts']].sum().add_suffix('_2_date')

train_test_2015 = pd.merge(train_test_2015, mc_string_2_date, left_on='claim_id', right_index=True, how='left')
del(mc_string, mc_string_2_date)

print('>>> mc string at last_non0_paid_yr')
mc_string = train_test[['claim_id', 'mc_string']][train_test.last_non0_paid_yr == train_test.paid_year]
mc_string_latest = mc_string[['claim_id']]
mc_string_latest['mc_string_counts_last'] = np.where(mc_string['mc_string'].isnull(), 0, (mc_string['mc_string'].str.count(' ')+1))
    
minor_cd = ['M25','LU7','MM2','MP8','I05','L79','MTF','MFC','MMO','L72','I02']
           
for cd in minor_cd:
    mc_string_latest[cd] = np.where(mc_string['mc_string'].str.contains(cd),1,0)
    
train_test_2015 = pd.merge(train_test_2015, mc_string_latest, on=['claim_id'], how='left')
train_test_2015['mc_string_counts_last_ratio'] = np.where(train_test_2015.mc_string_counts_2_date>0, train_test_2015.mc_string_counts_last/train_test_2015.mc_string_counts_2_date, np.nan)

del(mc_string, mc_string_latest)

print('>>> variables at last_non0_paid_yr')
last_non0_table = train_test[['claim_id', 'last_m', 'last_i', 'last_l', 'last_o','econ_unemployment_py', 'adj', 'paid_m', 'paid_i', 'paid_l', 'paid_o']][train_test.last_non0_paid_yr == train_test.paid_year].add_suffix('_last')
last_non0_table['paid_mi_last'] = last_non0_table['paid_m_last'] + last_non0_table['paid_i_last']
last_non0_table['paid_milo_last'] = last_non0_table['paid_m_last'] + last_non0_table['paid_i_last'] + last_non0_table['paid_l_last'] + last_non0_table['paid_o_last']
last_non0_table['last_milo_last'] = last_non0_table[['last_m_last','last_i_last','last_l_last','last_o_last']].max(axis=1)

train_test_2015 = pd.merge(train_test_2015, last_non0_table, left_on=['claim_id'], right_on = ['claim_id_last'], how='left')
train_test_2015.drop(['claim_id_last', 'paid_o_last', 'last_o_last',], 1, inplace=True)

del(last_non0_table)

train_test_2015['paid_m_last_2_date'] = np.where(train_test_2015.paid_m_2_date>0, np.maximum(train_test_2015['paid_m_last'], 0)/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['paid_i_last_2_date'] = np.where(train_test_2015.paid_i_2_date>0, np.maximum(train_test_2015['paid_i_last'], 0)/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['paid_l_last_2_date'] = np.where(train_test_2015.paid_l_2_date>0, np.maximum(train_test_2015['paid_l_last'], 0)/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['paid_mi_last_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, np.maximum(train_test_2015['paid_mi_last'], 0)/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['paid_milo_last_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, np.maximum(train_test_2015['paid_milo_last'], 0)/train_test_2015.paid_milo_2_date, np.nan)
   
train_test_2015.paid_m_last = np.log(np.maximum(train_test_2015.paid_m_last, 0) + 1)
train_test_2015.paid_i_last = np.log(np.maximum(train_test_2015.paid_i_last, 0) + 1)
train_test_2015.paid_l_last = np.log(np.maximum(train_test_2015.paid_l_last, 0) + 1)
train_test_2015.paid_mi_last = np.log(np.maximum(train_test_2015.paid_mi_last, 0) + 1)
train_test_2015.paid_milo_last = np.log(np.maximum(train_test_2015.paid_milo_last, 0) + 1)

print('>>> variables at cutoff_year')
cutoff_table = train_test[['claim_id', 'last_m', 'last_i', 'last_l', 'last_o', 'econ_unemployment_py', 'adj', 'paid_m', 'paid_i', 'paid_o', 'paid_l']][(2015 - train_test.slice) == train_test.paid_year].add_suffix('_cutoff')
cutoff_table['paid_mi_cutoff'] = cutoff_table['paid_m_cutoff'] + cutoff_table['paid_i_cutoff']
cutoff_table['paid_milo_cutoff'] = cutoff_table['paid_m_cutoff'] + cutoff_table['paid_i_cutoff'] + cutoff_table['paid_l_cutoff'] + cutoff_table['paid_o_cutoff']
cutoff_table['last_milo_cutoff'] = cutoff_table[['last_m_cutoff','last_i_cutoff','last_l_cutoff','last_o_cutoff']].max(axis=1)

train_test_2015 = pd.merge(train_test_2015, cutoff_table, left_on=['claim_id'], right_on = ['claim_id_cutoff'], how='left')
train_test_2015.drop(['claim_id_cutoff','paid_o_cutoff','last_o_cutoff',], 1, inplace=True)

del(cutoff_table)

train_test_2015['paid_m_cutoff_2_date'] = np.where(train_test_2015.paid_m_2_date>0, np.maximum(train_test_2015['paid_m_cutoff'], 0)/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['paid_i_cutoff_2_date'] = np.where(train_test_2015.paid_i_2_date>0, np.maximum(train_test_2015['paid_i_cutoff'], 0)/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['paid_l_cutoff_2_date'] = np.where(train_test_2015.paid_l_2_date>0, np.maximum(train_test_2015['paid_l_cutoff'], 0)/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['paid_mi_cutoff_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, np.maximum(train_test_2015['paid_mi_cutoff'], 0)/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['paid_milo_cutoff_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, np.maximum(train_test_2015['paid_milo_cutoff'], 0)/train_test_2015.paid_milo_2_date, np.nan)
   
train_test_2015.paid_m_cutoff = np.log(np.maximum(train_test_2015.paid_m_cutoff, 0) + 1)
train_test_2015.paid_i_cutoff = np.log(np.maximum(train_test_2015.paid_i_cutoff, 0) + 1)
train_test_2015.paid_l_cutoff = np.log(np.maximum(train_test_2015.paid_l_cutoff, 0) + 1)
train_test_2015.paid_mi_cutoff = np.log(np.maximum(train_test_2015.paid_mi_cutoff, 0) + 1)
train_test_2015.paid_milo_cutoff = np.log(np.maximum(train_test_2015.paid_milo_cutoff, 0) + 1)

#difference between last_non0_paid_yr and cutoff_yr
train_test_2015['diff_yr_last0_cutoff'] = (2015 - train_test_2015.slice) - train_test_2015.last_non0_paid_yr

train_test_2015.paid_m_2_date = np.log(train_test_2015.paid_m_2_date + 1)
train_test_2015.paid_i_2_date = np.log(train_test_2015.paid_i_2_date + 1)
train_test_2015.paid_mi_2_date = np.log(train_test_2015.paid_mi_2_date + 1)
train_test_2015.paid_l_2_date = np.log(train_test_2015.paid_l_2_date + 1)
train_test_2015.paid_milo_2_date = np.log(train_test_2015.paid_milo_2_date + 1)

train_test_2015.paid_m_2_date_max = np.log(np.maximum(train_test_2015.paid_m_2_date_max,0) + 1)
train_test_2015.paid_i_2_date_max = np.log(np.maximum(train_test_2015.paid_i_2_date_max,0) + 1)
train_test_2015.paid_mi_2_date_max = np.log(np.maximum(train_test_2015.paid_mi_2_date_max,0) + 1)
train_test_2015.paid_l_2_date_max = np.log(np.maximum(train_test_2015.paid_l_2_date_max,0) + 1)
train_test_2015.paid_milo_2_date_max = np.log(np.maximum(train_test_2015.paid_milo_2_date_max,0) + 1)

train_test_2015.paid_m_2_date_min = np.log(np.maximum(train_test_2015.paid_m_2_date_min,0) + 1)
train_test_2015.paid_i_2_date_min = np.log(np.maximum(train_test_2015.paid_i_2_date_min,0) + 1)
train_test_2015.paid_mi_2_date_min = np.log(np.maximum(train_test_2015.paid_mi_2_date_min,0) + 1)
train_test_2015.paid_l_2_date_min = np.log(np.maximum(train_test_2015.paid_l_2_date_min,0) + 1)
train_test_2015.paid_milo_2_date_min = np.log(np.maximum(train_test_2015.paid_milo_2_date_min,0) + 1)

train_test_2015.paid_m_2_date_mean = np.log(np.maximum(train_test_2015.paid_m_2_date_mean,0) + 1)
train_test_2015.paid_i_2_date_mean = np.log(np.maximum(train_test_2015.paid_i_2_date_mean,0) + 1)
train_test_2015.paid_mi_2_date_mean = np.log(np.maximum(train_test_2015.paid_mi_2_date_mean,0) + 1)
train_test_2015.paid_l_2_date_mean = np.log(np.maximum(train_test_2015.paid_l_2_date_mean,0) + 1)
train_test_2015.paid_milo_2_date_mean = np.log(np.maximum(train_test_2015.paid_milo_2_date_mean,0) + 1)

train_test_2015.last_non0_paid_yr = train_test_2015.last_non0_paid_yr.astype(str)
train_test_2015.last_non0_paid_yr[train_test_2015.last_non0_paid_yr == 'nan'] = '_M'

train_test_2015.cutoff_year = train_test_2015.cutoff_year.astype(str)
train_test_2015.cutoff_year[train_test_2015.cutoff_year == 'nan'] = '_M'

train_test_2015['cutoff_loss_yr'] = train_test_2015.cutoff_year + np.modf(train_test_2015.loss_yearmo)[1].astype(str)
train_test_2015['cutoff_last_non0_paid_yr'] = train_test_2015.cutoff_year + train_test_2015.last_non0_paid_yr

train_test_2015['last_m_last__last_non0_paid_yr'] = np.where(train_test_2015.last_m_last.isnull(), '_M', train_test_2015.last_m_last.astype(str)) + train_test_2015.last_non0_paid_yr
train_test_2015['last_i_last__last_non0_paid_yr'] = np.where(train_test_2015.last_i_last.isnull(), '_M', train_test_2015.last_i_last.astype(str)) + train_test_2015.last_non0_paid_yr
train_test_2015['last_milo_last__last_non0_paid_yr'] = np.where(train_test_2015.last_milo_last.isnull(), '_M', train_test_2015.last_milo_last.astype(str)) + train_test_2015.last_non0_paid_yr
train_test_2015['detri_icd9__last_non0_paid_yr'] = train_test_2015.detri_icd9 + train_test_2015.last_non0_paid_yr
train_test_2015['loss_year__last_non0_paid_yr'] = np.modf(train_test_2015.loss_yearmo)[1].astype(str) + train_test_2015.last_non0_paid_yr 

train_test_2015.adj_last[train_test_2015.adj_last.isnull()] = '_M'
train_test_2015['state_adj_last'] = train_test_2015.state + train_test_2015.adj_last
train_test_2015.adj_cutoff[train_test_2015.adj_cutoff.isnull()] = '_M'
train_test_2015['state_adj_cutoff'] = train_test_2015.state + train_test_2015.adj_cutoff

print('Done')

del(train_test)

others = ['paid_year']
econ = ['econ_gdp_py','econ_gdp_ar_py','econ_price_py','econ_price_allitems_py','econ_price_healthcare_py','econ_10yr_note_py','econ_unemployment_py']
fee = ['paid_i','paid_m','paid_l','paid_o','last_i','last_m','last_l','last_o','adj','mc_string','slice']
train_test_2015.drop(others + econ + fee, 1, inplace=True)

train_test_2015.shape

tv_id = pd.read_csv('./split.csv')
tv_id.head()
claim_id = tv_id.append(test_id)
claim_id.split.value_counts()

train_test_2015 = pd.merge(train_test_2015, claim_id, how='inner', on=['claim_id'])
train_test_2015.split.value_counts()

del(tv_id, claim_id, test_id)

cats = train_test_2015.dtypes[train_test_2015.dtypes=='object'].index
for var in cats:
    train_test_2015[var] = np.where(train_test_2015[var].isnull(), '_M', train_test_2015[var])
    train_test_2015[var+'_cnt'] = np.log(cnt(train_test_2015, [var])+1)
    
cats_var = ['clmnt_gender','initl_trtmt_cd','prexist_dsblty_in','catas_or_jntcvg_cd','state','suit_matter_type',\
        'diagnosis_icd9_cd','icd9_d1','icd9_d12','icd9_d123','cas_aia_cds_1_2','cas_aia_cds_3_4','clm_aia_cds_1_2','clm_aia_cds_3_4','occ_code',\
        'bpic','sic_cd','law_limit_tt','law_limit_pt','law_limit_pp','law_cola','law_offsets','law_ib_scheduled',\
        'state_wage','adj_last','adj_cutoff','cutoff_last_non0_paid_yr','last_m_last__last_non0_paid_yr',\
       'last_i_last__last_non0_paid_yr','loss_year__last_non0_paid_yr']



train_val = train_test_2015[train_test_2015.split.isin([0,1,2,3])]
train_val.drop(['split'],axis=1,inplace=True)
test = train_test_2015[train_test_2015.split.isin([-1])]

pred_test = np.zeros((train_test_2015[train_test_2015.split==-1].shape[0],21))
    
del(train_test_2015)

nfolds = 5
for fold in range(nfolds):
    print('fold {0}'.format(fold))
    train, val = train_test_split(train_val, train_size=.9, random_state=10*(fold+1))
    train['split'] = 1
    val['split'] = 2
    tv = train.append(val)
    
    train_test_p2015 = (tv.append(test)).reset_index(drop=True)
    
    cats = train_test_p2015.dtypes[train_test_p2015.dtypes=='object'].index
    for var in cats:
        train_test_p2015[var] = np.where(train_test_p2015[var].isnull(), '_M', train_test_p2015[var])
        train_test_p2015[var+'_loo'] = loo(train_test_p2015,[var],np.log(train_test_p2015.target+1),train_test_p2015.split==1,r_k=0.3)
    
    x_train = train_test_p2015[train_test_p2015.split==1].reset_index(drop=True)
    x_val = train_test_p2015[train_test_p2015.split==2].reset_index(drop=True)
    x_test = train_test_p2015[train_test_p2015.split==-1].reset_index(drop=True)

    bins = [-np.inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000,\
               94000, 133000, 189000, 268000, 381000, 540000, np.inf]
    bins_name = range(21)

    y_train = pd.cut(x_train.target, bins, labels=bins_name)
    y_val = pd.cut(x_val.target, bins, labels=bins_name)

    
    nt = x_train.shape[0]
    ntv = nt + x_val.shape[0]

    tv = x_train.append(x_val)
    train_test_p2015 = tv.append(x_test)

    del(x_train, x_val, x_test, tv)

    train_test_p2015.drop(['claim_id', 'target', 'split'], 1, inplace=True)

    ## Preprocessing and transforming to sparse data
    sparse_data = []

    for var in cats_var:
        dummy = pd.get_dummies(train_test_p2015[var].astype('category'))
        tmp = csr_matrix(dummy)
        sparse_data.append(tmp)

    nums = train_test_p2015.dtypes[train_test_p2015.dtypes!='object'].index
    for var in nums:
        if (sum(train_test_p2015[var].isnull()) > 0) | (sum(train_test_p2015[var] == -np.inf) > 0) | (sum(train_test_p2015[var] == np.inf) > 0):
            train_test_p2015[var][train_test_p2015[var].isnull()] = -1
            train_test_p2015[var][train_test_p2015[var] == -np.inf] = -1
            train_test_p2015[var][train_test_p2015[var] == np.inf] = -1
            
    scaler = StandardScaler()
    tmp = csr_matrix(scaler.fit_transform(train_test_p2015[nums]))
    sparse_data.append(tmp)

    ## sparse train and test data
    xtrain_test_p2015 = hstack(sparse_data, format = 'csr')
    xtv = xtrain_test_p2015[:ntv, :]
    x_test = xtrain_test_p2015[ntv:, :]

    x_train = xtv[:nt, :]
    x_val = xtv[nt:, :]

    del(xtrain_test_p2015, sparse_data, tmp, xtv)

    def nn_model():
        model = Sequential()

        model.add(Dense(200, input_dim = x_train.shape[1], init = 'normal'))
        model.add(PReLU())
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Dense(100, init = 'normal'))
        model.add(PReLU())
        model.add(BatchNormalization())    
        model.add(Dropout(0.2))

        model.add(Dense(21, init = 'normal', activation='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        return(model)

    nbags = 5
    nepochs = 25

    earlyStopping=EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')

    for j in range(nbags):
        print('bag {0}'.format(j))
        model = nn_model()
        fit = model.fit_generator(generator = batch_generator(x_train, y_train, 128, True),
                                  nb_epoch = nepochs,
                                  callbacks=[earlyStopping],
                                  samples_per_epoch = x_train.shape[0],
                                  validation_data=(x_val.todense(), y_val),
                                  verbose = 1)
    
        pred_test +=model.predict_generator(generator = batch_generatorp(x_test, 800, False), val_samples = x_test.shape[0])

pred_test /= nbags*nfolds


subm = pd.read_csv('/mnt/aigdata/OriginalData/sample_submission.csv', encoding='latin1')
print(subm.shape, pred_test.shape)
subm.ix[:,1:] = pred_test
subm.to_csv('./submission/'+ version + '.csv', index=False)









