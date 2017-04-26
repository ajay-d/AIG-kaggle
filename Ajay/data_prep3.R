rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)
library(xgboost)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

claims <- read_csv("data/claims.csv")
train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")
sample <- read_csv("data/sample_submission.csv")

claims.ids <- sort(unique(claims$claim_id))
train.ids <- sort(unique(train$claim_id))
test.ids <- sort(unique(test$claim_id))

test %>% 
  select(claim_id) %>%
  inner_join(train) %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  ungroup %>%
  count(slice, paid_year)

train %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  anti_join(test %>%
              select(claim_id)) %>%
  count(slice)

#####Shuffle slice#####

# claims.shuffle <- claims %>%
#   anti_join(test %>%
#               select(claim_id)) %>%
#   select(claim_id, slice)
# 
# set.seed(777)
# shuffle <- claims.shuffle[sample(nrow(claims.shuffle)), 'slice']
# 
# claims.shuffle <- data_frame(claim_id = claims.shuffle$claim_id,
#                              slice = shuffle$slice)
# 
# train.data <- train %>%
#   inner_join(claims.shuffle %>%
#                select(claim_id, slice)) %>%
#   filter(slice <= 10) %>%
#   select(claim_id, paid_year, slice, target)

#####No Shuffle slice#####

train.data <- train %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  filter(slice <= 10) %>%
  select(claim_id, paid_year, slice, target)


buckets <- c(-Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 
             47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)

#target
train.labels <- train.data %>%
  filter(paid_year == 2015) %>%
  select(claim_id, loss=target) %>%
  mutate(target = as.integer(cut(loss, buckets)) - 1)

##Combine both datasets
train.data <- train.data %>%
  select(claim_id, paid_year, slice) %>%
  mutate(cutoff_year = 2015-slice) %>%
  filter(paid_year <= cutoff_year)
test.data <- test %>%
  select(claim_id) %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  mutate(slice = slice-10) %>%
  inner_join(train %>%
               select(claim_id, paid_year)) %>%
  mutate(cutoff_year = 2015-slice)

train.data <- bind_rows(train.data, test.data)

train.data <- train.data %>%
  filter(paid_year <= cutoff_year) %>%
  inner_join(train %>%
               select(-target)) %>%
  mutate(any_i = paid_i > 0,
         any_m = paid_m > 0,
         any_l = paid_l > 0,
         any_o = paid_o > 0)

#last values
train.1 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  select(claim_id, econ_unemployment_py, adj, matches('econ|paid'))

#highest values values
train.2 <- train.data %>%
  group_by(claim_id) %>%
  summarise(any_i = max(any_i),
            any_m = max(any_m),
            any_l = max(any_l),
            any_o = max(any_o),
            mean_i = mean(paid_i),
            mean_m = mean(paid_m),
            mean_all = mean(paid_i+paid_m+paid_l+paid_o),
            max_prior = max(paid_i + paid_m))

date.vars <- names(claims) %>% str_subset("yearmo")
#date.vars <- setdiff(date.vars, c('abstract_yearmo', 'clm_create_yearmo', 'eff_yearmo', 'exp_yearmo'))

cat.vars <- c("clmnt_gender", "major_class_cd", "prexist_dsblty_in", "catas_or_jntcvg_cd",
              "suit_matter_type", "initl_trtmt_cd", "state",
              "occ_code", "sic_cd", "diagnosis_icd9_cd", "cas_aia_cds_1_2", "cas_aia_cds_3_4", "clm_aia_cds_1_2", "clm_aia_cds_3_4",
              "law_limit_tt", "law_limit_pt", "law_limit_pp", "law_cola", "law_offsets", "law_ib_scheduled",
              "dnb_spevnt_i", "dnb_rating", "dnb_comp_typ")

#cont.vars <- c("avg_wkly_wage", "imparmt_pct")
cont.vars <- c("avg_wkly_wage", "imparmt_pct",
               "dnb_emptotl", "dnb_emphere", "dnb_worth", "dnb_sales", "dnb_emphere",
               "econ_gdp_ly", "econ_gdp_ar_ly", "econ_price_ly", "econ_price_allitems_ly", "econ_price_healthcare_ly", "econ_10yr_note_ly", "econ_unemployment_ly")


claims %>% count(suit_matter_type)
claims %>% count(initl_trtmt_cd)
claims %>% count(prexist_dsblty_in)
claims %>% count(catas_or_jntcvg_cd)

claims %>% count(cas_aia_cds_1_2)
claims %>% count(sic_cd)
claims %>% count(major_class_cd)

claims %>% count(dnb_spevnt_i)
claims %>% count(dnb_rating)
claims %>% count(dnb_comp_typ)

#highest values values
train.3 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  select(claim_id, cutoff_year) %>%
  inner_join(claims %>%
               select(claim_id, one_of(cat.vars), one_of(cont.vars), one_of(date.vars))) %>%
  mutate(suit_yearmo = ifelse((cutoff_year - trunc(suit_yearmo)) < 0, NA, suit_yearmo),
         suit_matter_type = ifelse(is.na(suit_yearmo), NA, suit_matter_type),
         
         #suit_matter_type = as.numeric(suit_matter_type),
         occ_code = as.character(occ_code),
         initl_trtmt_cd = as.character(initl_trtmt_cd),
         
         imparmt_pct_yearmo = ifelse((cutoff_year - trunc(imparmt_pct_yearmo)) < 0, NA, imparmt_pct_yearmo),
         imparmt_pct = ifelse(is.na(imparmt_pct_yearmo), NA, imparmt_pct),
    
         mmi_yearmo = ifelse((cutoff_year - trunc(mmi_yearmo)) < 0, NA, mmi_yearmo),
    
         death_yearmo = ifelse((cutoff_year - trunc(death_yearmo)) < 0, NA, death_yearmo),
         death2_yearmo = ifelse((cutoff_year - trunc(death2_yearmo)) < 0, NA, death2_yearmo),
        
         surgery_yearmo = ifelse((cutoff_year - trunc(surgery_yearmo)) < 0, NA, surgery_yearmo),
         
         empl_hire_yearmo = ifelse((cutoff_year - trunc(empl_hire_yearmo)) < 0, NA, empl_hire_yearmo),
         rtrn_to_wrk_yearmo = ifelse((cutoff_year - trunc(rtrn_to_wrk_yearmo)) < 0, NA, rtrn_to_wrk_yearmo),
         
         reported_yearmo = ifelse((cutoff_year - trunc(reported_yearmo)) < 0, NA, reported_yearmo),
         abstract_yearmo = ifelse((cutoff_year - trunc(abstract_yearmo)) < 0, NA, abstract_yearmo),
         clm_create_yearmo = ifelse((cutoff_year - trunc(clm_create_yearmo)) < 0, NA, clm_create_yearmo),
         eff_yearmo = ifelse((cutoff_year - trunc(eff_yearmo)) < 0, NA, eff_yearmo),
         exp_yearmo = ifelse((cutoff_year - trunc(exp_yearmo)) < 0, NA, exp_yearmo),
         
         duration = exp_yearmo - eff_yearmo,
         report_to_claim = clm_create_yearmo - reported_yearmo,
         report_lag = reported_yearmo - loss_yearmo,
         eff_to_loss = eff_yearmo - loss_yearmo,
         loss_to_mmi = mmi_yearmo - loss_yearmo,
         loss_to_return = rtrn_to_wrk_yearmo - loss_yearmo,
         years_working = loss_yearmo - empl_hire_yearmo,
         
         clmnt_birth_yearmo = ifelse(clmnt_birth_yearmo > 9999, NA, clmnt_birth_yearmo),
         clmnt_birth_yearmo = ifelse(clmnt_birth_yearmo < 1901, NA, clmnt_birth_yearmo),
         
         age_at_injury = loss_yearmo - clmnt_birth_yearmo,
         age_at_hire = empl_hire_yearmo - clmnt_birth_yearmo,
         age_at_mmi = mmi_yearmo - clmnt_birth_yearmo,
         age_at_return = rtrn_to_wrk_yearmo - clmnt_birth_yearmo,
         
         cat_1 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd, state),
         cat_2 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd),
         cat_3 = paste0(diagnosis_icd9_cd, state),
         cat_4 = paste0(catas_or_jntcvg_cd, state)
         
         )


#checks
train.3 %>% 
  filter(age_at_injury < 0)
train.3 %>% 
  filter(loss_yearmo > cutoff_year)
train.3 %>% 
  filter(empl_hire_yearmo < clmnt_birth_yearmo)

train.data <- train.1 %>% 
  inner_join(train.2) %>%
  inner_join(train.3)

cat.vars <- union("adj", cat.vars)
cat.vars <- union(cat.vars, c('cat_1', 'cat_2', 'cat_3', 'cat_4'))
cat.table.long <- NULL
for(i in cat.vars) {
  
  train.var <- train.data %>%
    group_by_(i) %>%
    summarise(n.train = n()) %>%
    mutate(pct.train = n.train/sum(n.train))

  var.table <- train.var %>%
    rename_(value = i) %>%
    mutate(var = i)
  
  cat.table.long <- bind_rows(cat.table.long, var.table)
}
  

cat.table.long <- cat.table.long %>%
  group_by(var) %>%
  mutate(pct.total = n.train/sum(n.train)) %>%
  arrange(var, desc(n.train)) %>%
  mutate(#start at 0
         cat.number = row_number()-1)
#        cat.number = ifelse(pct.total < .01, NA, cat.number)) %>%
#  fill(cat.number)

train.recode <- train.data %>% select(claim_id)
for(i in cat.vars) {
  
  recode <- cat.table.long %>%
    ungroup() %>%
    filter(var == i) %>%
    select(value, cat.number) %>%
    setNames(c(i, 'cat.number'))
  
  df_train <- train.data %>%
    select_('claim_id', i) %>%
    inner_join(recode) %>%
    #drop old variable
    select(-2) %>%
    setNames(c('claim_id', i))
  
  train.recode <- train.recode %>%
    inner_join(df_train)

}



train.data <- train.data %>%
  select(-one_of(cat.vars)) %>%
  inner_join(train.recode) %>%
  ungroup()

####Box Transform####
names(train.data)
skew.vars <- setdiff(names(train.data), cat.vars)
skew.vars <- setdiff(skew.vars, skew.vars %>% str_subset("any|paid|mean"))
skew.vars

cont.vars <- setdiff(names(train.data), cat.vars)
cont.vars <- setdiff(cont.vars, cont.vars %>% str_subset("any"))

for (i in 2:length(train.data)) {
  nm <- names(train.data)[i]
  train.data[is.na(train.data[[nm]]), nm] <- median(train.data[[nm]], na.rm = TRUE)
  
  sk <- e1071::skewness(train.data[[nm]])
  if(sk > .25 & nm %in% skew.vars) {
    if(!is.na(caret::BoxCoxTrans(train.data[[nm]])$lambda)) {
      print(paste0(nm, ": Skew: ", sk))
      new_var <- forecast::BoxCox(train.data[[nm]], lambda = caret::BoxCoxTrans(train.data[[nm]])$lambda)
      train.data[[nm]] <- new_var
    }
  }
  
  ##Scale numeric data
  if(nm %in% cont.vars)
    train.data[[nm]] <- scale(train.data[[nm]])
}

####Prepare test data####
test.data <- train.data %>%
  filter(claim_id %in% test.ids)

train.data <- train.data %>%
  filter(claim_id %in% train.ids) %>%
  inner_join(train.labels)

set.seed(777)

#######################GBM 1########################################################

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 21,
              "nthread" = 12, 
              "eta" = 0.03,
              
              "alpha" = 3,
              "colsample_bytree" = 1,
              "gamma" = 5,
              "lambda" = 3,
              "max_depth" = 10,
              "min_child_weight" = 5,
              "subsample" = 1)

#CV all data
model.y <- train.data %>%
  use_series(target)

model.data <- train.data %>%
  select(-claim_id, -target, -loss) %>%
  data.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
bst.cv <- xgb.cv(params = param, 
                 data = xgbtrain, 
                 nrounds = 5000, 
                 nfold = 5,
                 early_stopping_rounds = 5,
                 print_every_n = 50)

bst.cv$best_iteration
bst.cv$best_ntreelimit
bst.cv$evaluation_log[['test_mlogloss_mean']][bst.cv$best_iteration]

round(bst.cv$best_iteration/.8)
bst.1 = xgb.train(params = param, 
                  data = xgbtrain, 
                  nrounds = round(bst.cv$best_iteration/.8),
                  print_every_n = 50)

# Compute feature importance matrix
importance_matrix_1 <- xgb.importance(colnames(xgbtrain), model = bst.1)

gbm.1 <- matrix(predict(bst.1, test.data %>% select(-claim_id) %>% data.matrix()), nrow=nrow(test.data), byrow=TRUE)
gbm.1 <- cbind(test.data$claim_id, gbm.1) %>%
  as.data.frame() %>%
  setNames(names(sample))

write.csv(gbm.1, gzfile("submission/gbm_slice3_cv1.csv.gz"), row.names=FALSE)

