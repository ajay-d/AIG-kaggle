create.dataset <- function(train.test){
  source('R/utils.r')
  
  
  train.test$last_adj <- as.integer(factor(train.test$last_adj,levels=c('Other','TTD','PPD','STLMT','PTD')))
  train.test$first_adj <- as.integer(factor(train.test$first_adj,levels=c('Other','TTD','PPD','STLMT','PTD')))
  
  train.test <- mutate(train.test,
                       
                       log_last_payment=log(last_payment),
                       log_first_payment=log(first_payment),
                       # slice = paid_year-cutoff_year,

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
                       # eff_yearmo,
                       # clmnt_birth_yearmo,clm_create_yearmo
                       # death_yearmo,death2_yearmo,
                       
                       # report_lag,
                       # surgery_report_lag,surgery_lag,
                       # imparmt_lag,imparmt_report_lag,
                       mmi_lag,mmi_report_lag,
                       # create_abstract_lag,
                       # abstract_lag,create_lag,
                       years_of_history,
                       
                       avg_wkly_wage,loss_age,clmnt_tenure,
                       # initl_trtmt_cd,
                       
                       # industry_int,industry_dnb_int,
                       catas_or_jntcvg_int,
                       # clmnt_gender_int,prexist_dsblty_int,
                       
                       exp_state,
                       exp_icd9,exp_major_code,exp_occ,exp_suit_type,
                       exp_law_tt,exp_law_pt,exp_law_pp,
                       law_limit_tt_int,law_limit_pt_int,law_limit_pp_int,
                       exp_cas_34,exp_clm_12,exp_clm_34,
                       exp_cas_1234,exp_cas_12,
                       exp_clm_1234,
                       exp_cas_clm_12,exp_cas_clm_34,exp_cas_12_clm_34,exp_cas_34_clm_12,
                       # exp_state_age,exp_gender_loss_age,
                       # exp_gender_age,
                       # exp_state_wage,exp_state_adj,
                       # exp_suit_year,exp_create_year,exp_loss_year,
                       # exp_state_loss,
                       
                       exp_suit_year_type,exp_state_suit_type,
                       exp_state_clm_34,
                       # exp_state_eff_year,
                       # exp_cutoff_last_year,exp_first_after_loss,exp_loss_year_first_year,
                       # exp_major_cd_suit_year,
                       # 
                       
                       # exp_state_clm_12,exp_state_cas_12,exp_state_suit_year,exp_suit_type_clm_3_4,
                       # exp_suit_type_cas_1_2,exp_suit_type_clm_1_2,exp_suit_year_cas_1_2,exp_suit_year_clm_1_2,
                       # exp_suit_year_clm_3_4,exp_init_code_clm_3_4,exp_cutoff_loss_year,
                       # exp_major_cd_state,exp_major_cd_suit_type,exp_major_cd_clm_12,exp_major_cd_clm_34,exp_law_ib_scheduled_clm_12,
                       
                       law_cola_int,law_offsets_int,law_ib_scheduled_int,

                       last_year,first_year,n_payment,
                       avg_payment,n_mean_mc,avg_payment_i,avg_payment_m,
                       total_payment,total_payment_i,total_payment_m,n_total_mc,last_before_cutoff,
                       # std_payment,std_payment_i,std_payment_m,
                       # min_payment,max_payment,min_payment_i,max_payment_i,min_payment_m,max_payment_m,
                       
                       last_i_last,last_m_last,last_l_last,
                       
                       last_3_payment_total,n_last_3_mc,last_3_payment_i_total,last_3_payment_m_total,
                       # last_5_payment_total,n_last_5_mc,last_5_payment_i_total,last_5_payment_m_total,
                       last_3_payment_avg,last_3_payment_i_avg,last_3_payment_m_avg,
                       # last_5_payment_avg,last_5_payment_i_avg,last_5_payment_m_avg,
                       
                       log_last_payment,last_adj,n_last_mc,
                       last_payment_i,last_payment_m,
                       log_first_payment,#first_adj,
                       # zero_precent,
                       
                       MP8,#LU7,M25,L79,MTF,MMQ,MM2,I02,II8,M28,L24,MMO,M26,
                       
                       # p2010i,p2011i,p2012i,p2013i,p2014i,
                       # p2010m,p2011m,p2012m,p2013m,p2014m,
                       # 
                       p0,p1,p2,#p3,p4,
                       p0i,p1i,p2i,#p3i,p4i,
                       p0m,p1m,p2m,#p3m,p4m,
                       
                       # dnb_rating,dnb_location_int,
                       # dnb_spevnt_i_int,dnb_comp_typ_int,dnb_history_int,dnb_conditn_int,
                       # dnb_finance_int,dnb_forowned_int,dnb_manuf_int,dnb_state_ph_int,
                       # dnb_publics_int,dnb_disbus_int,dnb_prescrn_int,dnb_frnchind_int,
                       
                       # gdp_cum,
                       # price_cum,price_allitem_cum,price_health_cum,
                       # gdp_cum_last,
                       # price_cum_last,price_allitem_cum_last,price_health_cum_last,
                       # gdp_cum_first,price_cum_first,price_allitem_cum_first,price_health_cum_first,
                       
                       # pay_last_gdp,pay_last_price,pay_last_price_allitem,pay_last_price_health,
                       # pay_avg_gdp,pay_avg_price,pay_avg_price_allitem,pay_avg_price_health,
                       
                       # last_3_bin_total,last_3_bin_avg,
                       # total_bin,avg_bin,
                       # avg_bin_cnt_0,avg_bin_cnt_1,
                       # avg_bin_cnt_2,avg_bin_cnt_3,avg_bin_cnt_4,
                       # avg_bin_cnt_5,avg_bin_cnt_6,avg_bin_cnt_7,
                       # avg_bin_cnt_8,avg_bin_cnt_9,avg_bin_cnt_10_up,
                       
                       econ_unemployment_py
  )
  
  return(train.test)
}