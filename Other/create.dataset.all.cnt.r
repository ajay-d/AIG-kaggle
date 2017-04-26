create.dataset.all <- function(train.test){
  source('R/utils.r')
  
  
  train.test$last_adj <- as.integer(factor(train.test$last_adj,levels=c('Other','TTD','PPD','STLMT','PTD')))
  train.test$first_adj <- as.integer(factor(train.test$first_adj,levels=c('Other','TTD','PPD','STLMT','PTD')))
  
  
  train.test <- mutate(train.test,
                       
                       log_last_payment=log(last_payment),
                       log_first_payment=log(first_payment),
                       # slice = paid_year-cutoff_year,

                       # censor time data by cutoff year
                       suit_yearmo = ifelse(cutoff_year<trunc(suit_yearmo), NA, suit_yearmo),
                       suit_matter_type = ifelse(is.na(suit_yearmo), NA, suit_matter_type),
                       imparmt_pct_yearmo = ifelse(cutoff_year<trunc(imparmt_pct_yearmo), NA, imparmt_pct_yearmo),
                       imparmt_pct = ifelse(is.na(imparmt_pct_yearmo), NA, imparmt_pct),
                       mmi_yearmo = ifelse(cutoff_year<trunc(mmi_yearmo), NA, mmi_yearmo),
                       surgery_yearmo = ifelse(cutoff_year<trunc(surgery_yearmo), NA, surgery_yearmo),
                       rtrn_to_wrk_yearmo = ifelse(cutoff_year<trunc(rtrn_to_wrk_yearmo), NA, rtrn_to_wrk_yearmo),
                       death_yearmo = ifelse(cutoff_year<trunc(death_yearmo), NA, death_yearmo),
                       death2_yearmo = ifelse(cutoff_year<trunc(death2_yearmo), NA, death2_yearmo),
                       abstract_yearmo = ifelse(cutoff_year<trunc(abstract_yearmo), NA, abstract_yearmo),
                       reported_yearmo = ifelse(cutoff_year<trunc(reported_yearmo), NA, reported_yearmo),
                       loss_yearmo = ifelse(cutoff_year<trunc(loss_yearmo), NA, loss_yearmo),
                       abstract_yearmo = ifelse(cutoff_year<trunc(abstract_yearmo), NA, abstract_yearmo),
                       empl_hire_yearmo = ifelse(cutoff_year<trunc(empl_hire_yearmo), NA, empl_hire_yearmo),
                       
                       surgery_lag = surgery_yearmo-loss_yearmo,
                       surgery_report_lag = surgery_yearmo-reported_yearmo,
                       report_lag = reported_yearmo-loss_yearmo,
                       loss_age = loss_yearmo-clmnt_birth_yearmo,
                       abstract_lag = abstract_yearmo-reported_yearmo,
                       create_lag = clm_create_yearmo-reported_yearmo,
                       imparmt_lag = imparmt_pct_yearmo-loss_yearmo,
                       imparmt_report_lag = imparmt_pct_yearmo-reported_yearmo,
                       clmnt_tenure = loss_yearmo-empl_hire_yearmo,
                       mmi_lag=mmi_yearmo-loss_yearmo,
                       mmi_report_lag=mmi_yearmo-reported_yearmo,
                       create_abstract_lag=clm_create_yearmo-abstract_yearmo,
                       years_of_history=cutoff_year-loss_yearmo,
                       first_after_loss=first_year-loss_yearmo,
                       
                       # economic variables
                       gdp_cum=log(econ_gdp_py/econ_gdp_ly),
                       price_cum=log(econ_price_py/econ_price_ly),
                       price_allitem_cum=log(econ_price_allitems_py/econ_price_allitems_ly),
                       price_health_cum=log(econ_price_healthcare_py/econ_price_healthcare_ly),
                       
                       gdp_cum_last=log(econ_gdp_py/last_gdp),
                       price_cum_last=log(econ_price_py/last_price),
                       price_allitem_cum_last=log(econ_price_allitems_py/last_price_allitems),
                       price_health_cum_last=log(econ_price_healthcare_py/last_price_healthcare),
                       
                       gdp_cum_first=log(econ_gdp_py/first_gdp),
                       price_cum_first=log(econ_price_py/first_price),
                       price_allitem_cum_first=log(econ_price_allitems_py/first_price_allitems),
                       price_health_cum_first=log(econ_price_healthcare_py/first_price_healthcare),
                       
                       pay_last_gdp=log_last_payment+gdp_cum_last,
                       pay_last_price=log_last_payment+price_cum_last,
                       pay_last_price_allitem=log_last_payment+price_allitem_cum_last,
                       pay_last_price_health=log_last_payment+price_health_cum_last
                       
                       # pay_avg_gdp=log(avg_pay_gdp*econ_gdp_py),
                       # pay_avg_price=log(avg_pay_price*econ_price_py),
                       # pay_avg_price_allitem=log(avg_pay_price_allitem*econ_price_allitems_py),
                       # pay_avg_price_health=log(avg_pay_price_healthcare*econ_price_allitems_py)
  )
  
  # select final list of variables
  train.test <- select(train.test,
                       claim_id,target,
                       cutoff_year,
                       # slice,
                       # paid_year,
                       
                       surgery_yearmo,imparmt_pct,imparmt_pct_yearmo,
                       mmi_yearmo,rtrn_to_wrk_yearmo,
                       suit_yearmo,loss_yearmo,
                       eff_yearmo,
                       clmnt_birth_yearmo,clm_create_yearmo,
                       death_yearmo,death2_yearmo,
                       
                       report_lag,
                       surgery_report_lag,surgery_lag,
                       imparmt_lag,imparmt_report_lag,
                       mmi_lag,mmi_report_lag,
                       create_abstract_lag,
                       abstract_lag,create_lag,
                       years_of_history,
                       
                       avg_wkly_wage,loss_age,clmnt_tenure,
                       initl_trtmt_cd,
                       
                       industry_int,industry_dnb_int,
                       catas_or_jntcvg_int,
                       clmnt_gender_int,prexist_dsblty_int,
                       
                       cnt_state,
                       cnt_icd9,cnt_major_code,cnt_occ,cnt_suit_type,
                       cnt_law_tt,cnt_law_pt,cnt_law_pp,
                       law_limit_tt_int,law_limit_pt_int,law_limit_pp_int,
                       cnt_cas_34,cnt_clm_12,cnt_clm_34,
                       cnt_cas_1234,cnt_cas_12,
                       cnt_clm_1234,
                       cnt_cas_clm_12,cnt_cas_clm_34,cnt_cas_12_clm_34,cnt_cas_34_clm_12,
                       cnt_state_age,cnt_gender_loss_age,
                       cnt_gender_age,
                       cnt_state_wage,cnt_state_adj,
                       cnt_suit_year,cnt_create_year,cnt_loss_year,
                       cnt_state_loss,
                       
                       cnt_suit_year_type,cnt_state_suit_type,
                       cnt_state_eff_year,cnt_state_clm_34,cnt_cutoff_last_year,
                       first_after_loss,cnt_loss_year_first_year,
                       cnt_major_cd_suit_year,
                       cnt_state_clm_12,cnt_state_cas_12,cnt_state_suit_year,cnt_suit_type_clm_3_4,
                       cnt_suit_type_cas_1_2,cnt_suit_type_clm_1_2,cnt_suit_year_cas_1_2,cnt_suit_year_clm_1_2,
                       cnt_suit_year_clm_3_4,cnt_init_code_clm_3_4,cnt_cutoff_loss_year,
                       cnt_major_cd_state,cnt_major_cd_suit_type,cnt_major_cd_clm_12,cnt_major_cd_clm_34,cnt_law_ib_scheduled_clm_12,
                       
                       ######## not yet test
                       # 
                       
                       law_cola_int,law_offsets_int,law_ib_scheduled_int,

                       last_year,first_year,n_payment,
                       avg_payment,n_mean_mc,avg_payment_i,avg_payment_m,
                       total_payment,total_payment_i,total_payment_m,n_total_mc,last_before_cutoff,
                       std_payment,std_payment_i,std_payment_m,
                       min_payment,max_payment,min_payment_i,max_payment_i,min_payment_m,max_payment_m,
                       
                       last_i_last,last_m_last,last_l_last,
                       last_3_payment_total,n_last_3_mc,last_3_payment_i_total,last_3_payment_m_total,
                       last_5_payment_total,n_last_5_mc,last_5_payment_i_total,last_5_payment_m_total,
                       last_3_payment_avg,last_3_payment_i_avg,last_3_payment_m_avg,
                       last_5_payment_avg,last_5_payment_i_avg,last_5_payment_m_avg,
                       
                       log_last_payment,last_adj,n_last_mc,
                       last_payment_i,last_payment_m,
                       log_first_payment,first_adj,
                       zero_precent,
                       
                       MP8,LU7,M25,L79,MTF,MMQ,MM2,I02,II8,M28,L24,MMO,M26,

                       p0,p1,p2,p3,p4,
                       p0i,p1i,p2i,p3i,p4i,
                       p0m,p1m,p2m,p3m,p4m,
                       
                       dnb_rating,dnb_location_int,
                       dnb_spevnt_i_int,dnb_comp_typ_int,dnb_history_int,dnb_conditn_int,
                       dnb_finance_int,dnb_manuf_int,dnb_state_ph_int,
                       dnb_disbus_int,dnb_prescrn_int,dnb_frnchind_int,
                       #dnb_forowned_int,dnb_publics_int,
                       
                       gdp_cum,
                       price_cum,
                       gdp_cum_last,
                       price_cum_last,
                       gdp_cum_first,price_cum_first,

                       
                       econ_unemployment_py
  )
  
  return(train.test)
}