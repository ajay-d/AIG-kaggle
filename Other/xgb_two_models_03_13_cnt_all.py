'''
based on xgb_two_models_03_13_cnt


'''
from __future__ import division
version = 'xgb_two_models_03_13_cnt_hold0.235513_all'


import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.utils import shuffle
from sklearn.metrics import log_loss
from sklearn.cross_validation import train_test_split
import operator

def cnt(dsn, varlist):
    df = dsn[varlist]
    df['cnt'] = 1.0
    return df.groupby(varlist).transform(np.sum).cnt
 

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

print('Done')

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

mi_paid['paid_m_non0'] = np.where(mi_paid.paid_m>0, 1, 0)
mi_paid['paid_i_non0'] = np.where(mi_paid.paid_i>0, 1, 0)
mi_paid['paid_l_non0'] = np.where(mi_paid.paid_l>0, 1, 0)
mi_paid['paid_o_non0'] = np.where(mi_paid.paid_o>0, 1, 0)
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
train_test_2015.paid_o_2_date = np.maximum(train_test_2015.paid_o_2_date, 0)
train_test_2015.paid_milo_2_date = np.maximum(train_test_2015.paid_milo_2_date, 0)

train_test_2015['paid_m_ratio'] = np.where(train_test_2015['paid_mi_2_date']>0, train_test_2015.paid_m_2_date/train_test_2015.paid_mi_2_date, 0)
train_test_2015['paid_mi_ratio'] = np.where(train_test_2015['paid_milo_2_date']>0, train_test_2015.paid_mi_2_date/train_test_2015.paid_milo_2_date, 0)

train_test_2015.paid_m_ratio = np.log(train_test_2015.paid_m_ratio + 1)
train_test_2015.paid_mi_ratio = np.log(train_test_2015.paid_mi_ratio + 1)

train_test_2015['paid_m_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_m_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_i_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_i_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_l_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_l_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_o_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_o_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_mi_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_mi_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)
train_test_2015['paid_milo_non0_2_date_prop'] = np.where(((2015 - train_test_2015.slice) - train_test_2015.first_paid_year)>0, train_test_2015.paid_milo_non0_2_date/((2015 - train_test_2015.slice) - train_test_2015.first_paid_year), -1)

print('>>> prior 10 medical and indemnity payment')
prior_10 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+10))]
prior_10['paid_mi'] = prior_10.paid_m + prior_10.paid_i
prior_10['paid_milo'] = prior_10.paid_m + prior_10.paid_i + prior_10.paid_l + prior_10.paid_o

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
train_test_2015.prior_10_sum_paid_o = np.log(np.maximum(train_test_2015.prior_10_sum_paid_o, 0) + 1)

train_test_2015.prior_10_max_paid_mi = np.log(np.maximum(train_test_2015.prior_10_max_paid_mi, 0) + 1)
train_test_2015.prior_10_max_paid_milo = np.log(np.maximum(train_test_2015.prior_10_max_paid_milo, 0) + 1)
train_test_2015.prior_10_max_paid_m = np.log(np.maximum(train_test_2015.prior_10_max_paid_m, 0) + 1)
train_test_2015.prior_10_max_paid_i = np.log(np.maximum(train_test_2015.prior_10_max_paid_i, 0) + 1)
train_test_2015.prior_10_max_paid_l = np.log(np.maximum(train_test_2015.prior_10_max_paid_l, 0) + 1)
train_test_2015.prior_10_max_paid_o = np.log(np.maximum(train_test_2015.prior_10_max_paid_o, 0) + 1)

train_test_2015.prior_10_min_paid_mi = np.log(np.minimum(train_test_2015.prior_10_min_paid_mi, 0) + 1)
train_test_2015.prior_10_min_paid_milo = np.log(np.minimum(train_test_2015.prior_10_min_paid_milo, 0) + 1)
train_test_2015.prior_10_min_paid_m = np.log(np.minimum(train_test_2015.prior_10_min_paid_m, 0) + 1)
train_test_2015.prior_10_min_paid_i = np.log(np.minimum(train_test_2015.prior_10_min_paid_i, 0) + 1)
train_test_2015.prior_10_min_paid_l = np.log(np.minimum(train_test_2015.prior_10_min_paid_l, 0) + 1)
train_test_2015.prior_10_min_paid_o = np.log(np.minimum(train_test_2015.prior_10_min_paid_o, 0) + 1)
 
train_test_2015.prior_10_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_10_mean_paid_mi, 0) + 1)
train_test_2015.prior_10_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_10_mean_paid_milo, 0) + 1)
train_test_2015.prior_10_mean_paid_m = np.log(np.maximum(train_test_2015.prior_10_mean_paid_m, 0) + 1)
train_test_2015.prior_10_mean_paid_i = np.log(np.maximum(train_test_2015.prior_10_mean_paid_i, 0) + 1)
train_test_2015.prior_10_mean_paid_l = np.log(np.maximum(train_test_2015.prior_10_mean_paid_l, 0) + 1)
train_test_2015.prior_10_mean_paid_o = np.log(np.maximum(train_test_2015.prior_10_mean_paid_o, 0) + 1)

print('>>> prior 5 medical and indemnity payment')
prior_5 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+5))]
prior_5['paid_mi'] = prior_5.paid_m + prior_5.paid_i
prior_5['paid_milo'] = prior_5.paid_m + prior_5.paid_i + prior_5.paid_l + prior_5.paid_o

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
train_test_2015.prior_5_sum_paid_o = np.log(np.maximum(train_test_2015.prior_5_sum_paid_o, 0) + 1)

train_test_2015.prior_5_max_paid_mi = np.log(np.maximum(train_test_2015.prior_5_max_paid_mi, 0) + 1)
train_test_2015.prior_5_max_paid_milo = np.log(np.maximum(train_test_2015.prior_5_max_paid_milo, 0) + 1)
train_test_2015.prior_5_max_paid_m = np.log(np.maximum(train_test_2015.prior_5_max_paid_m, 0) + 1)
train_test_2015.prior_5_max_paid_i = np.log(np.maximum(train_test_2015.prior_5_max_paid_i, 0) + 1)
train_test_2015.prior_5_max_paid_l = np.log(np.maximum(train_test_2015.prior_5_max_paid_l, 0) + 1)
train_test_2015.prior_5_max_paid_o = np.log(np.maximum(train_test_2015.prior_5_max_paid_o, 0) + 1)

train_test_2015.prior_5_min_paid_mi = np.log(np.minimum(train_test_2015.prior_5_min_paid_mi, 0) + 1)
train_test_2015.prior_5_min_paid_milo = np.log(np.minimum(train_test_2015.prior_5_min_paid_milo, 0) + 1)
train_test_2015.prior_5_min_paid_m = np.log(np.minimum(train_test_2015.prior_5_min_paid_m, 0) + 1)
train_test_2015.prior_5_min_paid_i = np.log(np.minimum(train_test_2015.prior_5_min_paid_i, 0) + 1)
train_test_2015.prior_5_min_paid_l = np.log(np.minimum(train_test_2015.prior_5_min_paid_l, 0) + 1)
train_test_2015.prior_5_min_paid_o = np.log(np.minimum(train_test_2015.prior_5_min_paid_o, 0) + 1)
 
train_test_2015.prior_5_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_5_mean_paid_mi, 0) + 1)
train_test_2015.prior_5_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_5_mean_paid_milo, 0) + 1)
train_test_2015.prior_5_mean_paid_m = np.log(np.maximum(train_test_2015.prior_5_mean_paid_m, 0) + 1)
train_test_2015.prior_5_mean_paid_i = np.log(np.maximum(train_test_2015.prior_5_mean_paid_i, 0) + 1)
train_test_2015.prior_5_mean_paid_l = np.log(np.maximum(train_test_2015.prior_5_mean_paid_l, 0) + 1)
train_test_2015.prior_5_mean_paid_o = np.log(np.maximum(train_test_2015.prior_5_mean_paid_o, 0) + 1)


print('>>> prior 3 medical and indemnity payment')
prior_3 = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o', 'adj']][((2015 - train_test.slice) >= train_test.paid_year) & ((2015 - train_test.slice) < (train_test.paid_year+3))]
prior_3['paid_mi'] = prior_3.paid_m + prior_3.paid_i
prior_3['paid_milo'] = prior_3.paid_m + prior_3.paid_i + prior_3.paid_l + prior_3.paid_o

for adju in ['Other','PPD','PTD','STLMT','TTD']:
    prior_3['adj_'+adju] = np.where(prior_3.adj.str.contains(adju),1,0)

prior_3.drop(['adj'],axis=1,inplace=True)    
prior_3_sum = prior_3.groupby('claim_id').sum().add_prefix('prior_3_sum_')
train_test_2015 = pd.merge(train_test_2015, prior_3_sum, left_on='claim_id', right_index=True, how='left')
del(prior_3_sum)

prior_3.drop(['adj_Other','adj_PPD','adj_PTD','adj_STLMT','adj_TTD',],axis=1,inplace=True) 
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

train_test_2015['prior_3_sum_paid_m_2_date'] = np.where(train_test_2015.paid_m_2_date>0, train_test_2015.prior_3_sum_paid_m/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['prior_3_sum_paid_i_2_date'] = np.where(train_test_2015.paid_i_2_date>0, train_test_2015.prior_3_sum_paid_i/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['prior_3_sum_paid_l_2_date'] = np.where(train_test_2015.paid_l_2_date>0, train_test_2015.prior_3_sum_paid_l/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['prior_3_sum_paid_o_2_date'] = np.where(train_test_2015.paid_o_2_date>0, train_test_2015.prior_3_sum_paid_o/train_test_2015.paid_o_2_date, np.nan)
train_test_2015['prior_3_sum_paid_mi_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, train_test_2015.prior_3_sum_paid_mi/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['prior_3_sum_paid_milo_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, train_test_2015.prior_3_sum_paid_milo/train_test_2015.paid_milo_2_date, np.nan)

train_test_2015.prior_3_sum_paid_mi = np.log(np.maximum(train_test_2015.prior_3_sum_paid_mi, 0) + 1)
train_test_2015.prior_3_sum_paid_milo = np.log(np.maximum(train_test_2015.prior_3_sum_paid_milo, 0) + 1)
train_test_2015.prior_3_sum_paid_m = np.log(np.maximum(train_test_2015.prior_3_sum_paid_m, 0) + 1)
train_test_2015.prior_3_sum_paid_i = np.log(np.maximum(train_test_2015.prior_3_sum_paid_i, 0) + 1)
train_test_2015.prior_3_sum_paid_l = np.log(np.maximum(train_test_2015.prior_3_sum_paid_l, 0) + 1)
train_test_2015.prior_3_sum_paid_o = np.log(np.maximum(train_test_2015.prior_3_sum_paid_o, 0) + 1)

train_test_2015.prior_3_max_paid_mi = np.log(np.maximum(train_test_2015.prior_3_max_paid_mi, 0) + 1)
train_test_2015.prior_3_max_paid_milo = np.log(np.maximum(train_test_2015.prior_3_max_paid_milo, 0) + 1)
train_test_2015.prior_3_max_paid_m = np.log(np.maximum(train_test_2015.prior_3_max_paid_m, 0) + 1)
train_test_2015.prior_3_max_paid_i = np.log(np.maximum(train_test_2015.prior_3_max_paid_i, 0) + 1)
train_test_2015.prior_3_max_paid_l = np.log(np.maximum(train_test_2015.prior_3_max_paid_l, 0) + 1)
train_test_2015.prior_3_max_paid_o = np.log(np.maximum(train_test_2015.prior_3_max_paid_o, 0) + 1)

train_test_2015.prior_3_min_paid_mi = np.log(np.minimum(train_test_2015.prior_3_min_paid_mi, 0) + 1)
train_test_2015.prior_3_min_paid_milo = np.log(np.minimum(train_test_2015.prior_3_min_paid_milo, 0) + 1)
train_test_2015.prior_3_min_paid_m = np.log(np.minimum(train_test_2015.prior_3_min_paid_m, 0) + 1)
train_test_2015.prior_3_min_paid_i = np.log(np.minimum(train_test_2015.prior_3_min_paid_i, 0) + 1)
train_test_2015.prior_3_min_paid_l = np.log(np.minimum(train_test_2015.prior_3_min_paid_l, 0) + 1)
train_test_2015.prior_3_min_paid_o = np.log(np.minimum(train_test_2015.prior_3_min_paid_o, 0) + 1)
 
train_test_2015.prior_3_mean_paid_mi = np.log(np.maximum(train_test_2015.prior_3_mean_paid_mi, 0) + 1)
train_test_2015.prior_3_mean_paid_milo = np.log(np.maximum(train_test_2015.prior_3_mean_paid_milo, 0) + 1)
train_test_2015.prior_3_mean_paid_m = np.log(np.maximum(train_test_2015.prior_3_mean_paid_m, 0) + 1)
train_test_2015.prior_3_mean_paid_i = np.log(np.maximum(train_test_2015.prior_3_mean_paid_i, 0) + 1)
train_test_2015.prior_3_mean_paid_l = np.log(np.maximum(train_test_2015.prior_3_mean_paid_l, 0) + 1)
train_test_2015.prior_3_mean_paid_o = np.log(np.maximum(train_test_2015.prior_3_mean_paid_o, 0) + 1)

print('>>> prior 1,2,3,4,5,6 medical and indemnity payment')

for i in [1,2,3,4,5,6]:
    prior = train_test[['claim_id', 'paid_m','paid_i', 'paid_l', 'paid_o','adj']][((2015 - train_test.slice) == (train_test.paid_year+i))]
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
    train_test_2015['paid_m_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_m_2_date>0, train_test_2015['paid_m_prior_'+str(i)]/train_test_2015.paid_m_2_date, np.nan)
    train_test_2015['paid_i_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_i_2_date>0, train_test_2015['paid_i_prior_'+str(i)]/train_test_2015.paid_i_2_date, np.nan)
    train_test_2015['paid_l_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_l_2_date>0, train_test_2015['paid_l_prior_'+str(i)]/train_test_2015.paid_l_2_date, np.nan)
    train_test_2015['paid_mi_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_mi_2_date>0, train_test_2015['paid_mi_prior_'+str(i)]/train_test_2015.paid_mi_2_date, np.nan)
    train_test_2015['paid_milo_2_date_prior_'+str(i)] = np.where(train_test_2015.paid_milo_2_date>0, train_test_2015['paid_milo_prior_'+str(i)]/train_test_2015.paid_milo_2_date, np.nan)
        
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

train_test_2015['paid_m_last_2_date'] = np.where(train_test_2015.paid_m_2_date>0, train_test_2015['paid_m_last']/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['paid_i_last_2_date'] = np.where(train_test_2015.paid_i_2_date>0, train_test_2015['paid_i_last']/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['paid_l_last_2_date'] = np.where(train_test_2015.paid_l_2_date>0, train_test_2015['paid_l_last']/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['paid_mi_last_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, train_test_2015['paid_mi_last']/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['paid_milo_last_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, train_test_2015['paid_milo_last']/train_test_2015.paid_milo_2_date, np.nan)
   
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

train_test_2015['paid_m_cutoff_2_date'] = np.where(train_test_2015.paid_m_2_date>0, train_test_2015['paid_m_cutoff']/train_test_2015.paid_m_2_date, np.nan)
train_test_2015['paid_i_cutoff_2_date'] = np.where(train_test_2015.paid_i_2_date>0, train_test_2015['paid_i_cutoff']/train_test_2015.paid_i_2_date, np.nan)
train_test_2015['paid_l_cutoff_2_date'] = np.where(train_test_2015.paid_l_2_date>0, train_test_2015['paid_l_cutoff']/train_test_2015.paid_l_2_date, np.nan)
train_test_2015['paid_mi_cutoff_2_date'] = np.where(train_test_2015.paid_mi_2_date>0, train_test_2015['paid_mi_cutoff']/train_test_2015.paid_mi_2_date, np.nan)
train_test_2015['paid_milo_cutoff_2_date'] = np.where(train_test_2015.paid_milo_2_date>0, train_test_2015['paid_milo_cutoff']/train_test_2015.paid_milo_2_date, np.nan)
   
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
train_test_2015.paid_o_2_date = np.log(train_test_2015.paid_o_2_date + 1)
train_test_2015.paid_milo_2_date = np.log(train_test_2015.paid_milo_2_date + 1)

train_test_2015.paid_m_2_date_max = np.log(np.maximum(train_test_2015.paid_m_2_date_max,0) + 1)
train_test_2015.paid_i_2_date_max = np.log(np.maximum(train_test_2015.paid_i_2_date_max,0) + 1)
train_test_2015.paid_mi_2_date_max = np.log(np.maximum(train_test_2015.paid_mi_2_date_max,0) + 1)
train_test_2015.paid_l_2_date_max = np.log(np.maximum(train_test_2015.paid_l_2_date_max,0) + 1)
train_test_2015.paid_o_2_date_max = np.log(np.maximum(train_test_2015.paid_o_2_date_max,0) + 1)
train_test_2015.paid_milo_2_date_max = np.log(np.maximum(train_test_2015.paid_milo_2_date_max,0) + 1)

train_test_2015.paid_m_2_date_min = np.log(np.maximum(train_test_2015.paid_m_2_date_min,0) + 1)
train_test_2015.paid_i_2_date_min = np.log(np.maximum(train_test_2015.paid_i_2_date_min,0) + 1)
train_test_2015.paid_mi_2_date_min = np.log(np.maximum(train_test_2015.paid_mi_2_date_min,0) + 1)
train_test_2015.paid_l_2_date_min = np.log(np.maximum(train_test_2015.paid_l_2_date_min,0) + 1)
train_test_2015.paid_o_2_date_min = np.log(np.maximum(train_test_2015.paid_o_2_date_min,0) + 1)
train_test_2015.paid_milo_2_date_min = np.log(np.maximum(train_test_2015.paid_milo_2_date_min,0) + 1)

train_test_2015.paid_m_2_date_mean = np.log(np.maximum(train_test_2015.paid_m_2_date_mean,0) + 1)
train_test_2015.paid_i_2_date_mean = np.log(np.maximum(train_test_2015.paid_i_2_date_mean,0) + 1)
train_test_2015.paid_mi_2_date_mean = np.log(np.maximum(train_test_2015.paid_mi_2_date_mean,0) + 1)
train_test_2015.paid_l_2_date_mean = np.log(np.maximum(train_test_2015.paid_l_2_date_mean,0) + 1)
train_test_2015.paid_o_2_date_mean = np.log(np.maximum(train_test_2015.paid_o_2_date_mean,0) + 1)
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


################################
# consider paid year 2015 only #
################################

del(train_test)

others = ['paid_year']
econ = ['econ_gdp_py','econ_gdp_ar_py','econ_price_py','econ_price_allitems_py','econ_price_healthcare_py','econ_10yr_note_py','econ_unemployment_py']
fee = ['paid_i','paid_m','paid_l','paid_o','last_i','last_m','last_l','last_o','adj','mc_string','slice']
train_test_2015.drop(others + econ + fee, 1, inplace=True)

tv_id = pd.read_csv('./split.csv')

tv_id.head()
claim_id = tv_id.append(test_id)
claim_id.split.value_counts()

train_test_2015 = pd.merge(train_test_2015, claim_id, how='inner', on=['claim_id'])
del(tv_id, claim_id, test_id)
train_test_2015.split.value_counts()


cats = train_test_2015.dtypes[train_test_2015.dtypes=='object'].index
for var in cats:
    train_test_2015[var] = np.where(train_test_2015[var].isnull(), '_M', train_test_2015[var])
    train_test_2015[var] = np.log(cnt(train_test_2015, [var])+1)


train_val = train_test_2015[train_test_2015.split.isin([0,1,2,3])]
train_val.drop(['split'],axis=1,inplace=True)
hold_test = train_test_2015[train_test_2015.split.isin([-1])]

pred_hold_combine = np.zeros((train_test_2015[train_test_2015.split==0].shape[0],21))
pred_test_combine = np.zeros((train_test_2015[train_test_2015.split==-1].shape[0],21))

nfolds = 10
for fold in range(nfolds):
    print('fold {0}'.format(fold))
    train, val = train_test_split(train_val, train_size=.9, random_state=10*(fold+1))
    train['split'] = 1
    val['split'] = 2
    tv = train.append(val)
    
    train_test_p2015 = (tv.append(hold_test)).reset_index(drop=True)

    x_train = train_test_p2015[train_test_p2015.split==1].reset_index(drop=True)
    x_val = train_test_p2015[train_test_p2015.split==2].reset_index(drop=True)
    x_test = train_test_p2015[train_test_p2015.split==-1].reset_index(drop=True)
         
    bins = [-np.inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000,\
               94000, 133000, 189000, 268000, 381000, 540000, np.inf]
    bins_name = range(21)
    
    y_train = pd.cut(x_train.target, bins, labels=bins_name)
    y_val = pd.cut(x_val.target, bins, labels=bins_name)
          
    train_target = x_train[['target']]
    val_target = x_val[['target']]
    
    x_train.drop(['claim_id', 'target', 'split'], 1, inplace=True)
    x_val.drop(['claim_id', 'target', 'split'], 1, inplace=True)
    x_test.drop(['claim_id', 'target', 'split'], 1, inplace=True)
        
    bins = [-np.inf, 0, np.inf]
    bins_name = range(2)
    
    y1_train = pd.cut(train_target.target, bins, labels=bins_name)
    y1_val = pd.cut(val_target.target, bins, labels=bins_name)
    
    d1train = xgb.DMatrix(x_train, label=y1_train, missing=np.nan)
    d1val = xgb.DMatrix(x_val, label=y1_val, missing=np.nan)
    dtest = xgb.DMatrix(x_test, missing=np.nan)
    
    pred1_test = np.zeros((x_test.shape[0], 2))
    
    params = {
        'objective':'multi:softprob',
        'num_class': 2,
        'eval_metric': 'mlogloss',
        'min_child_weight': 50,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 10,
        'subsample': 0.9,
        #'alpha': 0.5,
        'lambda':0.5,
        'gamma': 1,
        'silent': 1,
        'seed': 6688
    }
    
    watchlist = [(d1train, 'train'), (d1val, 'val')]      
    clf1 = xgb.train(params, d1train, 10000, watchlist, early_stopping_rounds=200, verbose_eval=10)
    
    pred1_test = clf1.predict(dtest,ntree_limit=clf1.best_ntree_limit).reshape(pred1_test.shape)
       
    bins = [0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000,\
               94000, 133000, 189000, 268000, 381000, 540000, np.inf]
    bins_name = range(20)
    
    y2_train = pd.cut(train_target.target[train_target.target>0], bins, labels=bins_name)
    y2_val = pd.cut(val_target.target[val_target.target>0], bins, labels=bins_name)
    y2_train.value_counts()
    
    x2_train = x_train[train_target.target>0]
    x2_val = x_val[val_target.target>0]
    print(x2_train.shape, x2_val.shape, y2_train.shape, y2_val.shape)
    
    d2train = xgb.DMatrix(x2_train, label=y2_train, missing=np.nan)
    d2val = xgb.DMatrix(x2_val, label=y2_val, missing=np.nan)
        
    pred2_test = np.zeros((x_test.shape[0], 20))
       
    params = {
        'objective':'multi:softprob',
        'num_class': 20,
        'eval_metric': 'mlogloss',
        'min_child_weight': 30,
        'eta': 0.01,
        'colsample_bytree': 0.5,
        'max_depth': 6,
        'subsample': 0.9,
        #'alpha': 1,
        #'lambda':0,
        'gamma': 3,
        'silent': 1,
        'seed': 6688
    }
    
    watchlist = [(d2train, 'train'), (d2val, 'val')]      
    clf2 = xgb.train(params, d2train, 10000, watchlist, early_stopping_rounds=200, verbose_eval=10)
    
    pred2_test = clf2.predict(dtest,ntree_limit=clf2.best_ntree_limit).reshape(pred2_test.shape)
    
    
    pred_test = np.zeros((x_test.shape[0], 21))
    
    pred_test[:,0] = pred1_test[:,0]
    
    for i in range(20):
        pred_test[:,i+1] = pred1_test[:,1]*pred2_test[:,i]
    
    
    pred_test_combine += pred_test
     
pred_test_combine /= nfolds
    
'''
combine cross validation results
'''
header = ['claim_id','Bucket_1','Bucket_2','Bucket_3','Bucket_4','Bucket_5','Bucket_6','Bucket_7','Bucket_8','Bucket_9','Bucket_10','Bucket_11','Bucket_12','Bucket_13','Bucket_14','Bucket_15','Bucket_16','Bucket_17','Bucket_18','Bucket_19','Bucket_20','Bucket_21']



subm = pd.read_csv('/mnt/aigdata/OriginalData/sample_submission.csv', encoding='latin1')
print(subm.shape, pred_test_combine.shape)
subm.ix[:,1:] = pred_test_combine
subm.to_csv('./submission/'+ version+ '.csv', index=False)


print('Done')


