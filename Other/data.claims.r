data.claims <- function(claims,train,payment.summary,buckets,slices,split,train.fold){
  
  library(dplyr)
  library(data.table)
  library(sqldf)
  
  strt <- Sys.time()
  source('R/utils.r')
  slice <- dplyr::slice
  
  # turn 2015 into actuall target
  # train$target <- as.integer(cut(train.3015$target, buckets)) - 1
  # try log
  train$target <- log(1+train$target)
  claims <- merge(claims,slices,all.x=TRUE,all.y=FALSE)
  claims <- mutate(claims,slice=ifelse(slice>10,slice-10,slice),
                         cutoff_year=2015-slice)
  
  # censor variable
  claims <- mutate(claims,
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
                   empl_hire_yearmo = ifelse(cutoff_year<trunc(empl_hire_yearmo), NA, empl_hire_yearmo)
                   )
  
  claims <- merge(claims,select(payment.summary,claim_id,first_year,last_year,last_adj),all.x=TRUE,all.y=FALSE)
  
  # treat claims
  # fix some data error
  claims$clmnt_birth_yearmo[claims$clmnt_birth_yearmo>2010] <- NA
  claims$empl_hire_yearmo[claims$empl_hire_yearmo>2016] <- NA
  claims$reported_yearmo[claims$reported_yearmo>2016] <- NA
  claims$reported_yearmo[claims$reported_yearmo<claims$loss_yearmo] <- NA
  claims$eff_yearmo[claims$eff_yearmo>2016] <- NA
  claims$avg_wkly_wage[claims$avg_wkly_wage>5000] <- NA
  
  # SIC to industry mapping
  claims$industry <- sic.to.industry(suppressWarnings(as.numeric(claims$sic_cd)))
  claims$industry_dnb <- sic.to.industry(suppressWarnings(as.numeric(claims$dnb_sic10)))
  
  # treat ordinal variables dnb_rating
  claims$dnb_rating[claims$dnb_rating==''] <- NA
  claims$dnb_rating[claims$dnb_rating=='--'] <- NA
  levels <- sort(unique(claims$dnb_rating)) 
  claims$dnb_rating <- as.integer(factor(claims$dnb_rating,levels=levels))
  
  # treat categorical variables
  claims <- mutate(claims,
                   clmnt_gender_int=char.to.int(clmnt_gender),
                   prexist_dsblty_int=char.to.int(prexist_dsblty_in),
                   catas_or_jntcvg_int=char.to.int(catas_or_jntcvg_cd),
                   law_cola_int=char.to.int(law_cola),
                   law_offsets_int=char.to.int(law_offsets),
                   law_ib_scheduled_int=char.to.int(law_ib_scheduled),
                   dnb_spevnt_i_int=char.to.int(dnb_spevnt_i),
                   dnb_comp_typ_int=char.to.int(dnb_comp_typ),
                   dnb_history_int=char.to.int(dnb_history),
                   dnb_conditn_int=char.to.int(dnb_conditn),
                   dnb_finance_int=char.to.int(dnb_finance),
                   dnb_forowned_int=char.to.int(dnb_forowned),
                   dnb_manuf_int=char.to.int(dnb_manuf),
                   dnb_state_ph_int=char.to.int(dnb_state_ph),
                   dnb_location_int=char.to.int(dnb_location),
                   dnb_publics_int=char.to.int(dnb_publics),
                   dnb_disbus_int=char.to.int(dnb_disbus),
                   dnb_prescrn_int=char.to.int(dnb_prescrn),
                   dnb_frnchind_int=char.to.int(dnb_frnchind),
                   
                   industry_int=char.to.int(industry),
                   industry_dnb_int=char.to.int(industry_dnb),
                   
                   law_limit_tt_int=char.to.int(law_limit_tt),
                   law_limit_pt_int=char.to.int(law_limit_pt),
                   law_limit_pp_int=char.to.int(law_limit_pp)
                   
  )
  
  
  # LOO
  set.seed(777)
  # train <- train %>% group_by(claim_id) %>% summarise(target=mean(target))
  
  claims <- merge(claims,select(train,claim_id,target),all=TRUE)
  claims <- merge(claims,split,all=TRUE)
  
  claims$split <- (!is.na(claims$target))&(claims$split%in%train.fold)
  claims$pred0 <- mean(claims$target[claims$split],na.rm=TRUE)
  claims$dummy <- 1
  claims$create_year <- floor(claims$clm_create_yearmo)
  claims$suit_year <- floor(claims$suit_yearmo)
  claims$eff_year <- floor(claims$eff_yearmo)
  claims$loss_year <- floor(claims$loss_yearmo)
  claims$loss_age <- floor(claims$loss_yearmo)-floor(claims$clmnt_birth_yearmo)
  claims$age <- floor(claims$clmnt_birth_yearmo)
  claims$wage <- floor(claims$avg_wkly_wage/500)
  
  claims$exp_icd9 <- my_exp1(claims, 'dummy', 'diagnosis_icd9_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_occ <- my_exp1(claims, 'dummy', 'occ_code', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_type <- my_exp1(claims, 'dummy', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_code <- my_exp1(claims, 'dummy', 'major_class_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_law_tt <- my_exp1(claims, 'dummy', 'law_limit_tt', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_law_pt <- my_exp1(claims, 'dummy', 'law_limit_pt', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_law_pp <- my_exp1(claims, 'dummy', 'law_limit_pp', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state <- my_exp1(claims, 'dummy', 'state', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_12 <- my_exp1(claims, 'dummy', 'cas_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_34 <- my_exp1(claims, 'dummy', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_clm_12 <- my_exp1(claims, 'dummy', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_clm_34 <- my_exp1(claims, 'dummy', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_1234 <- my_exp1(claims, 'cas_aia_cds_1_2', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_clm_1234 <- my_exp1(claims, 'clm_aia_cds_1_2', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_clm_12 <- my_exp1(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_clm_34 <- my_exp1(claims, 'clm_aia_cds_3_4', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_12_clm_34 <- my_exp1(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cas_34_clm_12 <- my_exp1(claims, 'cas_aia_cds_3_4', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a


  claims$exp_suit_year <- my_exp1(claims, 'dummy', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_create_year <- my_exp1(claims, 'dummy', 'create_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_loss_year <- my_exp1(claims, 'dummy', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_gender_loss_age <- my_exp1(claims, 'clmnt_gender', 'loss_age', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_gender_age <- my_exp1(claims, 'clmnt_gender', 'age', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_age <- my_exp1(claims, 'state', 'loss_age', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_wage <- my_exp1(claims, 'state', 'wage', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_loss <- my_exp1(claims, 'state', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_year_type <- my_exp1(claims, 'suit_year', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_suit_type <- my_exp1(claims, 'state', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_clm_34 <- my_exp1(claims, 'state', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_clm_12 <- my_exp1(claims, 'state', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_cas_12 <- my_exp1(claims, 'state', 'cas_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_suit_year <- my_exp1(claims, 'state', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_eff_year <- my_exp1(claims, 'state', 'eff_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_type_cas_1_2 <- my_exp1(claims, 'cas_aia_cds_1_2', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_type_clm_1_2 <- my_exp1(claims, 'clm_aia_cds_1_2', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_type_clm_3_4 <- my_exp1(claims, 'clm_aia_cds_3_4', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_year_cas_1_2 <- my_exp1(claims, 'cas_aia_cds_1_2', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_year_clm_1_2 <- my_exp1(claims, 'clm_aia_cds_1_2', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_suit_year_clm_3_4 <- my_exp1(claims, 'clm_aia_cds_3_4', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_init_code_clm_3_4 <- my_exp1(claims, 'clm_aia_cds_3_4', 'initl_trtmt_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a

  claims$exp_cutoff_loss_year <- my_exp1(claims, 'cutoff_year', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_cutoff_last_year <- my_exp1(claims, 'cutoff_year', 'last_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_loss_year_first_year <- my_exp1(claims, 'loss_year', 'first_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_cd_state <- my_exp1(claims, 'major_class_cd', 'state', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_cd_suit_type <- my_exp1(claims, 'major_class_cd', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_cd_suit_year <- my_exp1(claims, 'major_class_cd', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_cd_clm_12 <- my_exp1(claims, 'major_class_cd', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_major_cd_clm_34 <- my_exp1(claims, 'major_class_cd', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_state_adj <- my_exp1(claims, 'state', 'last_adj', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  claims$exp_law_ib_scheduled_clm_12 <- my_exp1(claims, 'law_ib_scheduled', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)$adj_a
  

  claims$cnt_icd9 <- my.f2cnt(claims, 'dummy', 'diagnosis_icd9_cd')
  claims$cnt_occ <- my.f2cnt(claims, 'dummy', 'occ_code')
  claims$cnt_suit_type <- my.f2cnt(claims, 'dummy', 'suit_matter_type')
  claims$cnt_major_code <- my.f2cnt(claims, 'dummy', 'major_class_cd')
  claims$cnt_law_tt <- my.f2cnt(claims, 'dummy', 'law_limit_tt')
  claims$cnt_law_pt <- my.f2cnt(claims, 'dummy', 'law_limit_pt')
  claims$cnt_law_pp <- my.f2cnt(claims, 'dummy', 'law_limit_pp')
  claims$cnt_state <- my.f2cnt(claims, 'dummy', 'state')
  claims$cnt_cas_12 <- my.f2cnt(claims, 'dummy', 'cas_aia_cds_1_2')
  claims$cnt_cas_34 <- my.f2cnt(claims, 'dummy', 'cas_aia_cds_3_4')
  claims$cnt_clm_12 <- my.f2cnt(claims, 'dummy', 'clm_aia_cds_1_2')
  claims$cnt_clm_34 <- my.f2cnt(claims, 'dummy', 'clm_aia_cds_3_4')
  claims$cnt_cas_1234 <- my.f2cnt(claims, 'cas_aia_cds_1_2', 'cas_aia_cds_3_4')
  claims$cnt_clm_1234 <- my.f2cnt(claims, 'clm_aia_cds_1_2', 'clm_aia_cds_3_4')
  claims$cnt_cas_clm_12 <- my.f2cnt(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_1_2')
  claims$cnt_cas_clm_34 <- my.f2cnt(claims, 'clm_aia_cds_3_4', 'cas_aia_cds_3_4')
  claims$cnt_cas_12_clm_34 <- my.f2cnt(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_3_4')
  claims$cnt_cas_34_clm_12 <- my.f2cnt(claims, 'cas_aia_cds_3_4', 'clm_aia_cds_1_2')

  claims$cnt_suit_year <- my.f2cnt(claims, 'dummy', 'suit_year')
  claims$cnt_create_year <- my.f2cnt(claims, 'dummy', 'create_year')
  claims$cnt_loss_year <- my.f2cnt(claims, 'dummy', 'loss_year')
  claims$cnt_gender_loss_age <- my.f2cnt(claims, 'clmnt_gender', 'loss_age')
  claims$cnt_gender_age <- my.f2cnt(claims, 'clmnt_gender', 'age')
  claims$cnt_state_age <- my.f2cnt(claims, 'state', 'loss_age')
  claims$cnt_state_wage <- my.f2cnt(claims, 'state', 'wage')
  claims$cnt_state_loss <- my.f2cnt(claims, 'state', 'loss_year')
  claims$cnt_suit_year_type <- my.f2cnt(claims, 'suit_year', 'suit_matter_type')
  claims$cnt_state_suit_type <- my.f2cnt(claims, 'state', 'suit_matter_type')
  claims$cnt_state_clm_34 <- my.f2cnt(claims, 'state', 'clm_aia_cds_3_4')
  claims$cnt_state_clm_12 <- my.f2cnt(claims, 'state', 'clm_aia_cds_1_2')
  claims$cnt_state_cas_12 <- my.f2cnt(claims, 'state', 'cas_aia_cds_1_2')
  claims$cnt_state_suit_year <- my.f2cnt(claims, 'state', 'suit_year')
  claims$cnt_state_eff_year <- my.f2cnt(claims, 'state', 'eff_year')
  claims$cnt_suit_type_cas_1_2 <- my.f2cnt(claims, 'cas_aia_cds_1_2', 'suit_matter_type')
  claims$cnt_suit_type_clm_1_2 <- my.f2cnt(claims, 'clm_aia_cds_1_2', 'suit_matter_type')
  claims$cnt_suit_type_clm_3_4 <- my.f2cnt(claims, 'clm_aia_cds_3_4', 'suit_matter_type')
  claims$cnt_suit_year_cas_1_2 <- my.f2cnt(claims, 'cas_aia_cds_1_2', 'suit_year')
  claims$cnt_suit_year_clm_1_2 <- my.f2cnt(claims, 'clm_aia_cds_1_2', 'suit_year')
  claims$cnt_suit_year_clm_3_4 <- my.f2cnt(claims, 'clm_aia_cds_3_4', 'suit_year')
  claims$cnt_init_code_clm_3_4 <- my.f2cnt(claims, 'clm_aia_cds_3_4', 'initl_trtmt_cd')

  claims$cnt_cutoff_loss_year <- my.f2cnt(claims, 'cutoff_year', 'loss_year')
  claims$cnt_cutoff_last_year <- my.f2cnt(claims, 'cutoff_year', 'last_year')
  claims$cnt_loss_year_first_year <- my.f2cnt(claims, 'loss_year', 'first_year')
  claims$cnt_major_cd_state <- my.f2cnt(claims, 'major_class_cd', 'state')
  claims$cnt_major_cd_suit_type <- my.f2cnt(claims, 'major_class_cd', 'suit_matter_type')
  claims$cnt_major_cd_suit_year <- my.f2cnt(claims, 'major_class_cd', 'suit_year')
  claims$cnt_major_cd_clm_12 <- my.f2cnt(claims, 'major_class_cd', 'clm_aia_cds_1_2')
  claims$cnt_major_cd_clm_34 <- my.f2cnt(claims, 'major_class_cd', 'clm_aia_cds_3_4')
  claims$cnt_state_adj <- my.f2cnt(claims, 'state', 'last_adj')
  claims$cnt_law_ib_scheduled_clm_12 <- my.f2cnt(claims, 'law_ib_scheduled', 'clm_aia_cds_1_2')

  claims$cat_icd9 <- my_cat1(claims, 'dummy', 'diagnosis_icd9_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_occ <- my_cat1(claims, 'dummy', 'occ_code', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_type <- my_cat1(claims, 'dummy', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_code <- my_cat1(claims, 'dummy', 'major_class_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_law_tt <- my_cat1(claims, 'dummy', 'law_limit_tt', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_law_pt <- my_cat1(claims, 'dummy', 'law_limit_pt', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_law_pp <- my_cat1(claims, 'dummy', 'law_limit_pp', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state <- my_cat1(claims, 'dummy', 'state', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_12 <- my_cat1(claims, 'dummy', 'cas_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_34 <- my_cat1(claims, 'dummy', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_clm_12 <- my_cat1(claims, 'dummy', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_clm_34 <- my_cat1(claims, 'dummy', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_1234 <- my_cat1(claims, 'cas_aia_cds_1_2', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_clm_1234 <- my_cat1(claims, 'clm_aia_cds_1_2', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_clm_12 <- my_cat1(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_clm_34 <- my_cat1(claims, 'clm_aia_cds_3_4', 'cas_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_12_clm_34 <- my_cat1(claims, 'cas_aia_cds_1_2', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cas_34_clm_12 <- my_cat1(claims, 'cas_aia_cds_3_4', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  
  claims$wage <- floor(claims$avg_wkly_wage/500)
  claims$cat_suit_year <- my_cat1(claims, 'dummy', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_create_year <- my_cat1(claims, 'dummy', 'create_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_loss_year <- my_cat1(claims, 'dummy', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_gender_loss_age <- my_cat1(claims, 'clmnt_gender', 'loss_age', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_gender_age <- my_cat1(claims, 'clmnt_gender', 'age', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_age <- my_cat1(claims, 'state', 'loss_age', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_wage <- my_cat1(claims, 'state', 'wage', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_loss <- my_cat1(claims, 'state', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_year_type <- my_cat1(claims, 'suit_year', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_suit_type <- my_cat1(claims, 'state', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_clm_34 <- my_cat1(claims, 'state', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_clm_12 <- my_cat1(claims, 'state', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_cas_12 <- my_cat1(claims, 'state', 'cas_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_suit_year <- my_cat1(claims, 'state', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_eff_year <- my_cat1(claims, 'state', 'eff_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_type_cas_1_2 <- my_cat1(claims, 'cas_aia_cds_1_2', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_type_clm_1_2 <- my_cat1(claims, 'clm_aia_cds_1_2', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_type_clm_3_4 <- my_cat1(claims, 'clm_aia_cds_3_4', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_year_cas_1_2 <- my_cat1(claims, 'cas_aia_cds_1_2', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_year_clm_1_2 <- my_cat1(claims, 'clm_aia_cds_1_2', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_suit_year_clm_3_4 <- my_cat1(claims, 'clm_aia_cds_3_4', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_init_code_clm_3_4 <- my_cat1(claims, 'clm_aia_cds_3_4', 'initl_trtmt_cd', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  
  claims$cat_cutoff_loss_year <- my_cat1(claims, 'cutoff_year', 'loss_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_cutoff_last_year <- my_cat1(claims, 'cutoff_year', 'last_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_loss_year_first_year <- my_cat1(claims, 'loss_year', 'first_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_cd_state <- my_cat1(claims, 'major_class_cd', 'state', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_cd_suit_type <- my_cat1(claims, 'major_class_cd', 'suit_matter_type', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_cd_suit_year <- my_cat1(claims, 'major_class_cd', 'suit_year', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_cd_clm_12 <- my_cat1(claims, 'major_class_cd', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_major_cd_clm_34 <- my_cat1(claims, 'major_class_cd', 'clm_aia_cds_3_4', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_state_adj <- my_cat1(claims, 'state', 'last_adj', 'target', 'pred0', claims$split, 5.0, r_k=.2)
  claims$cat_law_ib_scheduled_clm_12 <- my_cat1(claims, 'law_ib_scheduled', 'clm_aia_cds_1_2', 'target', 'pred0', claims$split, 5.0, r_k=.2)

  claims <- select(claims,-target,-dummy,-split,pred0,-cutoff_year,-slice,-first_year,-last_year,-last_adj)
  
  print(Sys.time()-strt)
  return(claims)
}