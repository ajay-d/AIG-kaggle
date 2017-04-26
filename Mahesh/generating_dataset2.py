import datetime
import numpy as np
import pandas as pd
import sys
import platform

#from joblib import Parallel, delayed
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import log_loss

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier


from xgboost import XGBClassifier


########
# DATA #
########

print("Read datasets...")
train = pd.read_csv("/mnt/aigdata/OriginalData/train.csv")
test = pd.read_csv("/mnt/aigdata/OriginalData/test.csv")
claims = pd.read_csv("/mnt/aigdata/OriginalData/claims.csv", encoding='ISO-8859-1')
print("Done with datasets...")

#################################
#Cleaning Train and Test datasets
#################################

### Creating bins for target and adding label to Train Dataset

bins = [-1, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000,10000000]
train['label'] = pd.cut(train['target'], bins, labels=np.arange(1,22), right = True)

### Dropping columns from Train not in Test to recreate Test conditions
train.drop(['paid_o', 'last_o' ], inplace = True, axis = 1)           #'paid_i', 'paid_m','paid_l', 'last_i', 'last_m', 'last_l',  'mc_string'


#, 'adj','target'

### Merging Train and Test sets
ttdata = train.append(test)


### Imputing mean to missing data in train + test - Note: OBJECT types screw up mean() and take super big memory
#col2imp = ttdata.columns[ttdata.dtypes!=object].values
col2imp = ['econ_10yr_note_py', 'econ_gdp_ar_py', 'econ_gdp_py',
       'econ_price_allitems_py', 'econ_price_healthcare_py',
       'econ_price_py', 'econ_unemployment_py']
ttdata[col2imp] = ttdata[col2imp].fillna(ttdata[col2imp].mean())

ttdata[col2imp] = np.log(1+ttdata[col2imp])
#The above imputation does not affect any values without NaNs

#KEEP ONLY RECORDS FROM TRAIN SET THAT MATCH SLICE

claims_temp = claims[['claim_id', 'slice']].copy()
ttdata = pd.merge(ttdata, claims_temp,  how='left', on='claim_id')

ttdata = (ttdata.reset_index()).drop('index', axis=1)
ttdata = ((ttdata.sort_values(['claim_id','paid_year'])).reset_index()).drop('index', axis=1)

ttdata['cutoffyr'] = np.where(ttdata['slice'] <11, 2015 - ttdata['slice'], 2015 - ttdata['slice'] +10)
ttdata['cutoffyr_flag'] = ttdata['cutoffyr'] - ttdata['paid_year']
ttdata.drop(['cutoffyr','slice'], inplace = True, axis = 1)

ttdata2015 = ttdata[ttdata['paid_year'] == 2015].copy()
ttdata2keep = ttdata[ttdata['cutoffyr_flag'] >= 0]

ttdata = ttdata2keep.append(ttdata2015)
ttdata = (ttdata.reset_index()).drop('index', axis=1)
ttdata = ((ttdata.sort_values(['claim_id','paid_year'])).reset_index()).drop('index', axis=1)

ttdata['cutoffyr_flag'] = ttdata['cutoffyr_flag']*-1

#CREATING FEATURES ---> TARGET_PREV, ADJ_PREV AND TARGET_AVERAGE (LAST 3 YEARS)
ttdata['target_prev'] = 0
ttdata['target_median3'] = 0
ttdata['adj_prev'] = ttdata['adj'].shift(1)

ttdata['i_prev'] = ttdata['paid_i'].shift(1)
ttdata['i_prev2'] = ttdata['paid_i'].shift(2)
ttdata['i_prev3'] = ttdata['paid_i'].shift(3)
ttdata['i_prev_avg3'] = 0

ttdata['m_prev'] = ttdata['paid_m'].shift(1)
ttdata['m_prev2'] = ttdata['paid_m'].shift(2)
ttdata['m_prev3'] = ttdata['paid_m'].shift(3)
ttdata['m_prev_avg3'] = 0

ttdata['target_shift1'] = ttdata['target'].shift(1)
ttdata['target_shift2'] = ttdata['target'].shift(2)
ttdata['target_shift3'] = ttdata['target'].shift(3)
temp = ttdata[['claim_id','paid_year']].groupby(['claim_id']).agg(['count'])
temp = (pd.DataFrame(temp.paid_year['count'])).reset_index()
ttdata = pd.merge(ttdata, temp,  how='left', on='claim_id')


ttdata['target_prev'] = ttdata['target_shift1']
ttdata['target_median3'] = np.where(ttdata['count']==2, ttdata['target_shift1'] , ttdata['target_median3'])
ttdata['target_median3'] = np.where(ttdata['count']==3, (ttdata['target_shift1']+ttdata['target_shift2'])/2 ,
                                    ttdata['target_median3'])
ttdata['target_median3'] = np.where(ttdata['count']>3, (ttdata['target_shift1']+ttdata['target_shift2']
                                                        +ttdata['target_shift3'])/3 ,
                                    ttdata['target_median3'])


ttdata['i_prev_avg3'] = np.where(ttdata['count']==2, ttdata['i_prev'] , ttdata['i_prev_avg3'])
ttdata['i_prev_avg3'] = np.where(ttdata['count']==3, (ttdata['i_prev']+ttdata['i_prev2'])/2 ,
                                    ttdata['i_prev_avg3'])
ttdata['i_prev_avg3'] = np.where(ttdata['count']>3, (ttdata['i_prev']+ttdata['i_prev2']
                                                        +ttdata['i_prev3'])/3 ,
                                    ttdata['i_prev_avg3'])


ttdata['m_prev_avg3'] = np.where(ttdata['count']==2, ttdata['m_prev'] , ttdata['m_prev_avg3'])
ttdata['m_prev_avg3'] = np.where(ttdata['count']==3, (ttdata['m_prev']+ttdata['m_prev2'])/2 ,
                                    ttdata['m_prev_avg3'])
ttdata['m_prev_avg3'] = np.where(ttdata['count']>3, (ttdata['m_prev']+ttdata['m_prev2']
                                                        +ttdata['m_prev3'])/3 ,
                                    ttdata['m_prev_avg3'])


#PREPROCESSING ON CLAIMS DATA

#REPLACING FEW FEATURES WITH COUNTS OF VALUES
foo2count = ['diagnosis_icd9_cd', 'sic_cd', 'occ_code', 'occ_desc1', 'occ_desc2'] #, 'cas_aia_cds_1_2', 'clm_aia_cds_1_2',
       #'clm_aia_cds_3_4',  'sic_cd', 'occ_code', 'occ_desc1', 'occ_desc2']

for x in foo2count:
    claims[x].fillna(0, inplace = True)
    claims[x] = claims[x].astype(str)
    foo= claims[x].value_counts(sort= True, dropna = True)
    freqfind = lambda y: foo[y]
    temp = np.log(claims[x].apply(lambda x: foo[x]))
    claims[x] = temp

#CREATING NEW AVG_WKLY_WAGE FEATURES
claims['aww1'] = np.log(1+claims['avg_wkly_wage']*2/3)
claims['aww2'] = np.log(1+4*claims['avg_wkly_wage']*2/3)
claims['aww3'] = np.log(1+12*4*claims['avg_wkly_wage']*2/3)
claims['tpd_amt'] = 0                                #TPD, PPD = impt_pct*avg_wkly_wage*2/3
claims['imparmt_pct'] = np.where(claims['imparmt_pct']>100, (claims['imparmt_pct'])/100, claims['imparmt_pct'])
claims['tpd_amt'] = np.where(claims['imparmt_pct'] >0, np.log(1+claims['imparmt_pct']*claims['avg_wkly_wage']*2/300), 
                            claims['tpd_amt'])

claims.avg_wkly_wage = np.log(1+claims.avg_wkly_wage) # LOG OF WEEKLY WAGE

#fooobj = claims.columns[claims.dtypes==object].values
fooobj = ['initl_trtmt_cd', 'prexist_dsblty_in', 'catas_or_jntcvg_cd', 'state', 'suit_matter_type',
'law_limit_tt', 'law_limit_pt', 'law_limit_pp', 'law_cola','law_offsets', 'law_ib_scheduled', 'clmnt_gender',
         
         'cas_aia_cds_1_2', 'clm_aia_cds_1_2','clm_aia_cds_3_4']

foorest = claims.columns[claims.dtypes!=object].values


### Applying Label Encoder to OBJECT columns with NaN values but leaving NaNs untouched. XGb can later handle NaNs directly.
for x in fooobj:
    claims[x].fillna(0, inplace = True)
    le = LabelEncoder()
    claims[x] = le.fit_transform(claims[x].astype(str))
    #claims[x][claims[x].notnull()] = le.fit_transform(claims[x][claims[x].notnull()].astype(str))


### Imputing mean values to the following columns NaN values
col2mean = ['econ_10yr_note_ly', 'econ_gdp_ar_ly', 'econ_gdp_ly',
       'econ_price_allitems_ly', 'econ_price_healthcare_ly', 'econ_price_ly',
       'econ_unemployment_ly']
claims[col2mean] = claims[col2mean].fillna(claims[col2mean].mean())
claims[col2mean] = np.log(1+claims[col2mean])

#>>>>>*****this works! --- col2impNA = list(set(foorest) - set(col2mean))


claims['surgery_yearmo_flt'] =  claims['surgery_yearmo'].copy()
claims['imparmt_pct_yearmo_flt'] =  claims['imparmt_pct_yearmo'].copy()
claims['clmnt_birth_yearmo_flt'] =  claims['clmnt_birth_yearmo'].copy()
claims['empl_hire_yearmo_flt'] =  claims['empl_hire_yearmo'].copy()
claims['death_yearmo_flt'] =  claims['death_yearmo'].copy()
claims['death2_yearmo_flt'] =  claims['death2_yearmo'].copy()
claims['loss_yearmo_flt'] =  claims['loss_yearmo'].copy()
claims['reported_yearmo_flt'] =  claims['reported_yearmo'].copy()
claims['abstract_yearmo_flt'] =  claims['abstract_yearmo'].copy()
claims['clm_create_yearmo_flt'] =  claims['clm_create_yearmo'].copy()
claims['mmi_yearmo_flt'] =  claims['mmi_yearmo'].copy()
claims['rtrn_to_wrk_yearmo_flt'] =  claims['rtrn_to_wrk_yearmo'].copy()
claims['eff_yearmo_flt'] =  claims['eff_yearmo'].copy()
claims['exp_yearmo_flt'] =  claims['exp_yearmo'].copy()
claims['suit_yearmo_flt'] =  claims['suit_yearmo'].copy()

#MERGING TRAIN AND CLAIMS DATASETS

# TO MAKE CODE RUN FAST, we only want paid_year = 2015 ; comment it later
ttdata = ttdata[ttdata.paid_year == 2015]

### Join Train Test data with Claims
data = pd.merge(ttdata, claims,  how='left', on='claim_id')

# delete old data frames to reuse space
del train, test, claims, ttdata

### Truncate dates
col2dround = ['surgery_yearmo', 'imparmt_pct_yearmo', 
       'clmnt_birth_yearmo', 'empl_hire_yearmo', 'death_yearmo',
       'death2_yearmo', 'loss_yearmo', 'reported_yearmo', 'abstract_yearmo',
       'clm_create_yearmo', 'mmi_yearmo', 'rtrn_to_wrk_yearmo', 'eff_yearmo',
       'exp_yearmo', 'suit_yearmo']

for x in col2dround:
    data[x][data[x].notnull()] = data[x][data[x].notnull()].astype(int)
    

#cutting of train data AFTER cutoff year according to slice
data['slice'] = np.where(data['slice']>10, data['slice'] - 10, data['slice'])
data['cutoff_yr'] = data['paid_year'] - data['slice']

col2cut = ['suit_yearmo','imparmt_pct_yearmo','mmi_yearmo','rtrn_to_wrk_yearmo','death_yearmo','death2_yearmo',
          'surgery_yearmo']

for x in col2cut:
    data[x].fillna(0, inplace = True)
    data[x] = np.where(((data['cutoff_yr'] - data[x])<0),0,data[x])

data['suit_matter_type'][data['suit_yearmo'] == 0] = 0
data['imparmt_pct'][data['imparmt_pct_yearmo'] == 0] = 0 
data['tpd_amt'][data['imparmt_pct_yearmo'] == 0] = 0 



#LIMITING AND ENCODING DATES 
col2limit=  ['surgery_yearmo','imparmt_pct_yearmo', 'clmnt_birth_yearmo', 
             'empl_hire_yearmo', 'death_yearmo', 'loss_yearmo',
       'reported_yearmo', 'mmi_yearmo', 'rtrn_to_wrk_yearmo', 
       'suit_yearmo', 'cutoff_yr']

for x in col2limit:
    data[x].fillna(0, inplace = True)
    data[x] = np.where(data[x]<0, 0, data[x])
    data[x] = np.where((data[x]>0) & (data[x] < 1980), 1980, data[x])
    #data[x] = np.log(1+data[x])
    le = LabelEncoder()
    data[x] = le.fit_transform(data[x].astype(str))
    

#CREATING NEW FEATURES - DIFFERENCE BETWEEN DATES

#DIFFERENCE between REPORTED AND LOSS YEAR - Months
data['reported_yearmo_flt'][data['reported_yearmo_flt']<data['loss_yearmo_flt']] = data['loss_yearmo_flt']
data['reported_yearmo_flt'][data['reported_yearmo_flt']>2016] = data['loss_yearmo_flt']   #Cleaning reported year

data['diff_repo_loss'] = (data['reported_yearmo_flt']-data['loss_yearmo_flt'])*12
data['diff_repo_loss'] = data['diff_repo_loss'].round(0).astype(int)

#DIFFERENCE BETWEEN IMPAIRMENT YEAR AND LOSS YEAR - Months
data['imparmt_pct_yearmo_flt'].fillna(0, inplace = True)
data['diff_impmt_loss'] = 0
data['diff_impmt_loss'][data['imparmt_pct_yearmo_flt']!=0] = (data['imparmt_pct_yearmo_flt'] - data['loss_yearmo_flt'])*12
#data['diff_mmi_repo'] = np.where(data['diff_mmi_repo']<0, 0, data['diff_mmi_repo'])
data['diff_impmt_loss'] = data['diff_impmt_loss'].round(0).astype(int)

data['diff_impmt_repo'] = 0
data['diff_impmt_repo'][data['imparmt_pct_yearmo_flt']!=0] = (data['imparmt_pct_yearmo_flt'] - data['reported_yearmo_flt'])*12
#data['diff_mmi_repo'] = np.where(data['diff_mmi_repo']<0, 0, data['diff_mmi_repo'])
data['diff_impmt_repo'] = data['diff_impmt_repo'].round(0).astype(int)

#DIFFERENCE BETWEEN MMI AND LOSS YEAR - Months
data['mmi_yearmo_flt'].fillna(0, inplace = True)
data['diff_mmi_loss'] = 0
data['diff_mmi_loss'][data['mmi_yearmo_flt']!=0] = (data['mmi_yearmo_flt'] - data['loss_yearmo_flt'])*12
#data['diff_mmi_repo'] = np.where(data['diff_mmi_repo']<0, 0, data['diff_mmi_repo'])
data['diff_mmi_loss'] = data['diff_mmi_loss'].round(0).astype(int)


data['diff_mmi_repo'] = 0
data['diff_mmi_repo'][data['mmi_yearmo_flt']!=0] = (data['mmi_yearmo_flt'] - data['reported_yearmo_flt'])*12
#data['diff_mmi_repo'] = np.where(data['diff_mmi_repo']<0, 0, data['diff_mmi_repo'])
data['diff_mmi_repo'] = data['diff_mmi_repo'].round(0).astype(int)

#MERGING DEATH COLUMNS
data['death_yearmo_flt'] = np.where(data['death_yearmo_flt'].isnull(), data['death2_yearmo_flt'], data['death_yearmo_flt'])

#DIFFERENCE BETWEEN DEATH YEAR AND LOSS YEAR - Months
data['death_yearmo_flt'].fillna(0, inplace = True)
data['diff_death_loss'] = 0
data['diff_death_loss'][data['death_yearmo_flt']!=0] = (data['death_yearmo_flt'] - data['loss_yearmo_flt'])*12
data['diff_death_loss'] = data['diff_death_loss'].round(0).astype(int)

data['diff_death_repo'] = 0
data['diff_death_repo'][data['death_yearmo_flt']!=0] = (data['death_yearmo_flt'] - data['reported_yearmo_flt'])*12
data['diff_death_repo'] = data['diff_death_repo'].round(0).astype(int)

#DIFFERENCE BETWEEN RETURN TO WORK YEAR AND LOSS YEAR - Months
data['rtrn_to_wrk_yearmo_flt'].fillna(0, inplace = True)
data['diff_return_loss'] = 0
data['diff_return_loss'][data['rtrn_to_wrk_yearmo_flt']!=0] = (data['rtrn_to_wrk_yearmo_flt'] - data['loss_yearmo_flt'])*12
data['diff_return_loss'] = data['diff_return_loss'].round(0).astype(int)


data['diff_return_repo'] = 0
data['diff_return_repo'][data['rtrn_to_wrk_yearmo_flt']!=0] = (data['rtrn_to_wrk_yearmo_flt'] - data['reported_yearmo_flt'])*12
data['diff_return_repo'] = data['diff_return_repo'].round(0).astype(int)

#DIFFERENCE BETWEEN SUIT YEAR AND LOSS YEAR - Months
data['suit_yearmo_flt'].fillna(0, inplace = True)
data['diff_suit_loss'] = 0
data['diff_suit_loss'][data['suit_yearmo_flt']!=0] = (data['suit_yearmo_flt'] - data['loss_yearmo_flt'])*12
data['diff_suit_loss'] = data['diff_suit_loss'].round(0).astype(int)

data['diff_suit_repo'] = 0
data['diff_suit_repo'][data['suit_yearmo_flt']!=0] = (data['suit_yearmo_flt'] - data['reported_yearmo_flt'])*12
data['diff_suit_repo'] = data['diff_suit_repo'].round(0).astype(int)

####################################################################################
####################################################################################

data['paid_year'] = data['paid_year']+0.92

#DIFFERENCE BETWEEN PAID YEAR AND REPORTED YEAR - Months
data['diff_paid_repo'] = (data['paid_year'] - data['reported_yearmo_flt'])*12
data['diff_paid_repo'] = data['diff_paid_repo'].round(0).astype(int)
#difference between paid year and reported year - buckets - new features
data['lp1'] = 0
data['lp1'] = np.where((data['paid_year'] - data['reported_yearmo_flt'])<=2, 1, data['lp1'])   # 0-2 yrs   

data['lp1'] = np.where(((data['paid_year'] - data['reported_yearmo_flt'])>2) &  ((data['paid_year'] - data['reported_yearmo_flt'])<=4), 2, data['lp1'])   # 2-4 yrs    

data['lp1'] = np.where(((data['paid_year'] - data['reported_yearmo_flt'])>4) & ((data['paid_year'] - data['reported_yearmo_flt'])<=6), 3, data['lp1'])   # 4-6 yrs     

data['lp1'] = np.where((data['paid_year'] - data['reported_yearmo_flt'])>6, 4, data['lp1'])   # >6 yrs  





data['diff_paid_loss'] = (data['paid_year'] - data['loss_yearmo_flt'])*12
data['diff_paid_loss'] = data['diff_paid_loss'].round(0).astype(int)


#DIFFERENCE BETWEEN IMPAIRMENT YEAR AND PAID YEAR - Months
data['diff_paid_impmt'] = 0
data['diff_paid_impmt'][data['imparmt_pct_yearmo_flt']!=0] = (data['paid_year'] - data['imparmt_pct_yearmo_flt'])*12
#claims_foo['diff_mmi_repo'] = np.where(claims_foo['diff_mmi_repo']<0, 0, claims_foo['diff_mmi_repo'])
data['diff_paid_impmt'] = data['diff_paid_repo'].round(0).astype(int)


#DIFFERENCE BETWEEN MMI AND PAID YEAR - Months
data['diff_paid_mmi'] = 0
data['diff_paid_mmi'][data['mmi_yearmo_flt']!=0] = (data['paid_year'] - data['mmi_yearmo_flt'])*12
#claims_foo['diff_mmi_repo'] = np.where(claims_foo['diff_mmi_repo']<0, 0, claims_foo['diff_mmi_repo'])
data['diff_paid_mmi'] = data['diff_paid_mmi'].round(0).astype(int)


#DIFFERENCE BETWEEN DEATH YEAR AND PAID YEAR - Months
data['diff_paid_death'] = 0
data['diff_paid_death'][data['death_yearmo_flt']!=0] = (data['paid_year'] - data['death_yearmo_flt'])*12
#claims_foo['diff_mmi_repo'] = np.where(claims_foo['diff_mmi_repo']<0, 0, claims_foo['diff_mmi_repo'])
data['diff_paid_death'] = data['diff_paid_death'].round(0).astype(int)


#DIFFERENCE BETWEEN RETURN TO WORK YEAR AND PAID YEAR - Months
data['diff_paid_return'] = 0
data['diff_paid_return'][data['rtrn_to_wrk_yearmo_flt']!=0] = (data['paid_year'] - data['rtrn_to_wrk_yearmo_flt'])*12
#claims_foo['diff_mmi_repo'] = np.where(claims_foo['diff_mmi_repo']<0, 0, claims_foo['diff_mmi_repo'])
data['diff_paid_return'] = data['diff_paid_return'].round(0).astype(int)


#DIFFERENCE BETWEEN SUIT YEAR AND PAID YEAR - Months
data['diff_paid_suit'] = 0
data['diff_paid_suit'][data['suit_yearmo_flt']!=0] = (data['paid_year'] - data['suit_yearmo_flt'])*12
#claims_foo['diff_mmi_repo'] = np.where(claims_foo['diff_mmi_repo']<0, 0, claims_foo['diff_mmi_repo'])
data['diff_paid_suit'] = data['diff_paid_suit'].round(0).astype(int)


#AGE OF CLAIMANT - Years
data['clmnt_birth_yearmo_flt'].fillna(1965, inplace= True)
data['age_claimaint'] = (data['paid_year'] - data['clmnt_birth_yearmo_flt']).round(0).astype(int)
#data['age_claimaint'] = np.where(data['age_claimaint']<18, data['paid_year'] - data['loss_yearmo_flt'] - 35, data['age_claimaint'])
data['age_claimaint'] = np.where(data['age_claimaint']>100, 100, data['age_claimaint'])

#TIME TO RETIREMENT from PAID YEAR - Years
#data['time2retire'] = 0
#data['time2retire'][data['law_limit_pt2'].isin(['retirement'])] = 67 - data['age_claimaint'] # x12?
#data['time2retire'] = np.where(data['time2retire']<0, 0, data['time2retire'])

#data.drop(['reported_yearmo','imparmt_pct_yearmo','mmi_yearmo',
 #          'death_yearmo','rtrn_to_wrk_yearmo','suit_yearmo','loss_yearmo','law_limit_pt2',
#         'clmnt_birth_yearmo'], inplace = True, axis = 1)


#CONVERTING GENERATED FEATURES TO LOGs
ncol2log = ['diff_repo_loss','diff_impmt_loss','diff_mmi_loss','diff_death_loss','diff_return_loss','diff_suit_loss', 
'diff_paid_repo','diff_paid_impmt','diff_paid_mmi','diff_paid_death','diff_paid_return','diff_paid_suit', 'age_claimaint',
 
'diff_impmt_repo', 'diff_mmi_repo','diff_death_repo','diff_return_repo','diff_suit_repo','diff_paid_loss',
           
'target_prev', 'target_median3']

for x in ncol2log:
    data[x] = np.where(data[x]<0, 0, data[x])
    data[x] = np.log(1+data[x])

#adj_prev
le = LabelEncoder()
data['adj_prev'] = le.fit_transform(data['adj_prev'])


#CONVERTING I_PREV AND M_PREV TO LOGS
for x in ['i_prev', 'm_prev', 'i_prev_avg3', 'm_prev_avg3']:
    data[x] = np.where( data[x] <0 , 0 , data[x])
    data[x] = np.log(1+data[x])

####################
# SPLIT TRAIN TEST #
####################


train = data[data['label'].notnull()]
final_test = data[data['label'].isnull()]
#del data

labels = train['label']
train.drop(['claim_id','paid_year','label'], inplace = True, axis = 1)
final_test.drop(['claim_id','label'], inplace = True, axis = 1)
             
# Split training data into training and validation sets
#X_train, X_test, y_train, y_test = train_test_split(train, labels, test_size=0.2, random_state=21)

train.fillna(0, inplace = True)
final_test.fillna(0, inplace =True)

#KEEPING ONLY FEW FEATURES FOR TRAINING AND TESTING -- Score LB 26200 CV 25723 with these fields

#ORIGINAL
init_trainfoo = 0

if init_trainfoo:
    train_foo = train[['econ_unemployment_py', 'slice', 'diagnosis_icd9_cd', 'surgery_yearmo',
       'clmnt_gender', 'avg_wkly_wage', 'cas_aia_cds_1_2', 'clm_aia_cds_1_2',
       'clm_aia_cds_3_4', 'imparmt_pct_yearmo', 'imparmt_pct',
       'initl_trtmt_cd', 'prexist_dsblty_in', 'catas_or_jntcvg_cd', 'sic_cd',
       'clmnt_birth_yearmo', 'empl_hire_yearmo', 'death_yearmo', 'loss_yearmo',
       'reported_yearmo', 'mmi_yearmo',
       'rtrn_to_wrk_yearmo', 'state', 'occ_code',
       'occ_desc1', 'occ_desc2', 'suit_yearmo', 'suit_matter_type',
       'law_limit_tt', 'law_limit_pt', 'law_limit_pp', 'law_cola',
       'law_offsets', 'law_ib_scheduled','econ_gdp_ly', 'econ_gdp_ar_ly', 'econ_price_ly',
       'econ_price_allitems_ly', 'econ_price_healthcare_ly',
       'econ_10yr_note_ly', 'econ_unemployment_ly', 'cutoff_yr',
        
        'diff_repo_loss','diff_impmt_loss','diff_mmi_loss','diff_death_loss','diff_return_loss','diff_suit_loss', 
        'diff_paid_repo', 'diff_paid_impmt','diff_paid_mmi','diff_paid_death','diff_paid_return','diff_paid_suit', 
        'age_claimaint',
                  
        'target_prev',  'target_median3','adj_prev','cutoffyr_flag']]

               
#'diff_impmt_repo', 'diff_mmi_repo','diff_death_repo','diff_return_repo','diff_suit_repo','diff_paid_loss'

#TRYING AFTER A RUN OF RANDOM FOREST   #CV 0.24181 LB 0.25128  LB 0.24807

updated_trainfoo = 1  #after multiple runs of Random Forest/XGB to determine important features

if updated_trainfoo:    
    train_foo = train[['econ_unemployment_py', 'diagnosis_icd9_cd', 'surgery_yearmo', 'slice', #
        'aww1', 'cas_aia_cds_1_2', 'clm_aia_cds_1_2', 'clmnt_gender',#'clmnt_gender', #replace avg_wkly_wage with comp_max
       'clm_aia_cds_3_4', 'imparmt_pct_yearmo', 'imparmt_pct',
       'initl_trtmt_cd',  'catas_or_jntcvg_cd',#  'clmnt_birth_yearmo', #'sic_cd', 
       'loss_yearmo','reported_yearmo', 'death_yearmo',#'empl_hire_yearmo',
       'mmi_yearmo','state', 'occ_desc1', 'rtrn_to_wrk_yearmo',#'occ_code', 'rtrn_to_wrk_yearmo', 
       'suit_yearmo', 'suit_matter_type', 'law_limit_tt',#'occ_desc2', 
       'law_limit_pt', 'law_limit_pp', 'law_cola', 'law_offsets',
       'law_ib_scheduled', 
        #'econ_gdp_ly', 'econ_gdp_ar_ly', 'econ_price_ly',
       #'econ_price_allitems_ly', 'econ_price_healthcare_ly',
       #'econ_10yr_note_ly', 'econ_unemployment_ly', 
        #'cutoff_yr',
       'diff_repo_loss', 'diff_impmt_loss', 'diff_mmi_loss', 'diff_suit_loss',
       'diff_paid_repo', 'diff_paid_impmt', 'diff_paid_mmi', 'diff_paid_death',
       'diff_paid_return', 'diff_paid_suit', 'age_claimaint', 'target_prev',
       'target_median3', 'adj_prev', 'cutoffyr_flag', 
        
        'lp1', #'lp2', 'lp3', 'lp4',  #paidyr reportedyr difference buckets
        
        'i_prev', 'm_prev','i_prev_avg3', 'm_prev_avg3', 'tpd_amt'
        ]]
    
test_foo = final_test[train_foo.columns.values]
#Comp_max #0.2
#claims['aww1'] = np.log(1+claims['avg_wkly_wage']*2/3)
#claims['aww2'] = np.log(1+4*claims['avg_wkly_wage']*2/3) #0.24081
#claims['aww3'] = np.log(1+12*4*claims['avg_wkly_wage']*2/3)

final_test_foo = final_test[train_foo.columns.values]
train_foo.to_pickle("train_foo_feat2.pkl")
test_foo.to_pickle("final_test_foo_feat2.pkl")
labels.to_pickle('labels_foo.pkl')

#SELECTING FEATURES THROUGH RANDOM FOREST
feature_selection = 0

if feature_selection == 1:
    from sklearn.ensemble import RandomForestClassifier

    model= RandomForestClassifier(n_estimators=600, max_depth = 20, min_samples_leaf = 5, n_jobs=-1, verbose=True)
    model.fit(train_foo, labels)
    
    #pd.options.display.max_rows = 999
    cols = train_foo.columns

    feat_imp = pd.DataFrame( cols, model.feature_importances_)

    feat_test = feat_imp[feat_imp.index.values > 0.001]
    feat_test = feat_test.reset_index()
    feat_test.drop('index', inplace = True, axis = 1)

    train_foo = train_foo[feat_test[0].values]