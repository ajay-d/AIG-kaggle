rm(list = ls(all = TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(broom)
library(ggplot2)
library(stringr)
library(xgboost)
library(feather)
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


buckets <- c( - Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000,
             47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)

#target
train.labels <- train.data %>%
  filter(paid_year == 2015) %>%
  select(claim_id, loss = target) %>%
  mutate(target = as.integer(cut(loss, buckets)) - 1)

train.labels %>%
  count(target) %>%
  mutate(pct = n/sum(n)) %>%
  arrange(target) %>%
  as.data.frame()

##Combine both datasets
train.data <- train.data %>%
  select(claim_id, paid_year, slice) %>%
  mutate(cutoff_year = 2015 - slice) %>%
  filter(paid_year <= cutoff_year)
test.data <- test %>%
  select(claim_id) %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  mutate(slice = slice - 10) %>%
  inner_join(train %>%
               select(claim_id, paid_year)) %>%
  mutate(cutoff_year = 2015 - slice)

train.data <- bind_rows(train.data, test.data)

train.data <- train.data %>%
  filter(paid_year <= cutoff_year) %>%
  inner_join(train %>%
               select( - target)) %>%
  mutate(any_i = paid_i > 0,
         any_m = paid_m > 0,
         any_l = paid_l > 0,
         any_o = paid_o > 0)

##Regress values
reg.all <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  mutate(paid_im = paid_i + paid_m) %>%
  select(claim_id, slice, paid_i, paid_m, paid_im)

y1 <- train.data %>%
  group_by(claim_id) %>%
  select(claim_id, paid_year, slice, cutoff_year, y = paid_i)

x1 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year < max(paid_year)) %>%
  mutate(x = lag(paid_i)) %>%
  filter(!is.na(x)) %>%
  select(claim_id, paid_year, slice, cutoff_year, x)

reg1 <- x1 %>%
  inner_join(y1) %>%
  group_by(claim_id) %>%
  do(tidy(lm(y ~ x, .))) %>%
  filter(term == 'x')

y2 <- train.data %>%
  group_by(claim_id) %>%
  select(claim_id, paid_year, slice, cutoff_year, y = paid_m)

x2 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year < max(paid_year)) %>%
  mutate(x = lag(paid_m)) %>%
  filter(!is.na(x)) %>%
  select(claim_id, paid_year, slice, cutoff_year, x)

reg2 <- x2 %>%
  inner_join(y2) %>%
  group_by(claim_id) %>%
  do(tidy(lm(y ~ x, .))) %>%
  filter(term == 'x')

y3 <- train.data %>%
  group_by(claim_id) %>%
  mutate(y = paid_i + paid_m) %>%
  select(claim_id, paid_year, slice, cutoff_year, y)

x3 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year < max(paid_year)) %>%
  mutate(x = lag(paid_m) + lag(paid_i)) %>%
  filter(!is.na(x)) %>%
  select(claim_id, paid_year, slice, cutoff_year, x)

reg3 <- x3 %>%
  inner_join(y3) %>%
  group_by(claim_id) %>%
  do(tidy(lm(y ~ x, .))) %>%
  filter(term == 'x')

reg.all <- reg.all %>%
  left_join(reg1 %>%
              select(claim_id, reg_i = estimate)) %>%
  left_join(reg2 %>%
              select(claim_id, reg_m = estimate)) %>%
  left_join(reg3 %>%
              select(claim_id, reg_im = estimate)) %>%
  replace_na(list(reg_i = 0, reg_m = 0, reg_im = 0)) %>%
  mutate(reg_i_scale = reg_i * slice,
         reg_m_scale = reg_m * slice,
         reg_im_scale = reg_im * slice,
         reg_i_scale_sq = reg_i ** slice,
         reg_m_scale_sq = reg_m ** slice,
         reg_im_scale_sq = reg_im ** slice,
         paid_im_scale1 = paid_im * reg_im_scale,
         paid_im_scale2 = paid_i * reg_i_scale + paid_m * reg_m_scale,
         paid_im_scale1_sq = paid_im * reg_im_scale_sq,
         paid_im_scale2_sq = paid_i * reg_i_scale_sq + paid_m * reg_m_scale_sq)


#last values
train.1 <- train.data %>%
  group_by(claim_id) %>%
  arrange(claim_id, paid_year) %>%
  mutate(pct_paid_i = paid_i / (lag(paid_i) + 1),
         pct_paid_m = paid_m / (lag(paid_m) + 1),
         pct_paid_im = (paid_i + paid_m) / (lag(paid_i) + lag(paid_m) + 1),
         sign_paid_i = sign(paid_i - lag(paid_i, default = 0)),
         sign_paid_m = sign(paid_m - lag(paid_m, default = 0)),
         sign_paid_im = sign(paid_i + paid_m - lag(paid_i, default = 0) - lag(paid_m, default = 0))) %>%
  filter(paid_year == max(paid_year)) %>%
  select(claim_id, econ_unemployment_py, adj, matches('econ|paid'), slice) %>%
  mutate(pct_paid_i = ifelse(is.na(pct_paid_i) | is.nan(pct_paid_i) | is.infinite(pct_paid_i), 0, pct_paid_i),
         pct_paid_m = ifelse(is.na(pct_paid_m) | is.nan(pct_paid_m) | is.infinite(pct_paid_m), 0, pct_paid_m),
         pct_paid_im = ifelse(is.na(pct_paid_im) | is.nan(pct_paid_im) | is.infinite(pct_paid_im), 0, pct_paid_im),
         pct_paid_i_scale = pct_paid_i * slice,
         pct_paid_m_scale = pct_paid_m * slice,
         pct_paid_im_scale = pct_paid_im * slice,

         pct_paid_i_scale_sq = pct_paid_i ** slice,
         pct_paid_m_scale_sq = pct_paid_m ** slice,
         pct_paid_im_scale_sq = pct_paid_im ** slice,

         pct_paid_i_scale_dollar = pct_paid_i_scale * paid_i,
         pct_paid_m_scale_dollar = pct_paid_m_scale * paid_m,
         pct_paid_im_scale_dollar = pct_paid_im_scale * (paid_i + paid_m),

         pct_paid_i_scale_sq_dollar = pct_paid_i_scale_sq * paid_i,
         pct_paid_m_scale_sq_dollar = pct_paid_m_scale_sq * paid_m,
         pct_paid_im_scale_sq_dollar = pct_paid_im_scale_sq * (paid_i + paid_m))


train.2 <- train.data %>%
  group_by(claim_id) %>%
  summarise(any_i = max(any_i),
            any_m = max(any_m),
            any_l = max(any_l),
            any_o = max(any_o),

            mean_i = mean(paid_i),
            mean_m = mean(paid_m),
            mean_l = mean(paid_l),
            mean_o = mean(paid_o),
            mean_im = mean(paid_i + paid_m),
            mean_all = mean(paid_i + paid_m + paid_l + paid_o),

            min_i = min(paid_i),
            min_m = min(paid_m),
            min_l = min(paid_l),
            min_o = min(paid_o),
            min_im = min(paid_i + paid_m),
            min_all = min(paid_i + paid_m + paid_l + paid_o),

            max_i = max(paid_i),
            max_m = max(paid_m),
            max_l = max(paid_l),
            max_o = max(paid_o),
            max_im = max(paid_i + paid_m),
            max_all = max(paid_i + paid_m + paid_l + paid_o),

            med_i = median(paid_i),
            med_m = median(paid_m),
            med_l = median(paid_l),
            med_o = median(paid_o),
            med_im = median(paid_i + paid_m),
            med_all = median(paid_i + paid_m + paid_l + paid_o),

            n_hist = n()
            )

train.2a <- train.data %>%
  group_by(claim_id, slice) %>%
  arrange(claim_id, paid_year) %>%
  mutate(pct_paid_i = paid_i / lag(paid_i)) %>%
  filter(!is.na(pct_paid_i), !is.infinite(pct_paid_i)) %>%
  summarise(pct_paid_i_ave = mean(pct_paid_i))

train.2b <- train.data %>%
  group_by(claim_id, slice) %>%
  arrange(claim_id, paid_year) %>%
  mutate(pct_paid_m = paid_m / lag(paid_m)) %>%
  filter(!is.na(pct_paid_m), !is.infinite(pct_paid_m)) %>%
  summarise(pct_paid_m_ave = mean(pct_paid_m))

train.2c <- train.data %>%
  group_by(claim_id, slice) %>%
  arrange(claim_id, paid_year) %>%
  mutate(pct_paid_i = paid_i / lag(paid_i),
         pct_paid_m = paid_m / lag(paid_m),
         pct_paid_im = (paid_i + paid_m) / (lag(paid_i) + lag(paid_m))) %>%
  filter(!is.na(pct_paid_im), !is.infinite(pct_paid_im)) %>%
  summarise(pct_paid_im_ave = mean(pct_paid_im))

train.2d <- train.1 %>%
  select(claim_id, slice, paid_i, paid_m) %>%
  full_join(train.2a) %>%
  full_join(train.2b) %>%
  full_join(train.2c) %>%
  replace_na(list(pct_paid_i_ave = 0, pct_paid_m_ave = 0, pct_paid_im_ave = 0)) %>%
  mutate(pct_paid_i_ave_scale = pct_paid_i_ave * slice,
         pct_paid_m_ave_scale = pct_paid_m_ave * slice,
         pct_paid_im_ave_scale = pct_paid_im_ave * slice,

         pct_paid_i_ave_scale_sq = pct_paid_i_ave ** slice,
         pct_paid_m_ave_scale_sq = pct_paid_m_ave ** slice,
         pct_paid_im_ave_scale_sq = pct_paid_im_ave ** slice,

         pct_paid_i_ave_scale_dollar = pct_paid_i_ave_scale * paid_i,
         pct_paid_m_ave_scale_dollar = pct_paid_m_ave_scale * paid_m,
         pct_paid_im_ave_scale_dollar = pct_paid_im_ave_scale * (paid_i + paid_m),

         pct_paid_i_ave_scale_sq_dollar = pct_paid_i_ave_scale_sq * paid_i,
         pct_paid_m_ave_scale_sq_dollar = pct_paid_m_ave_scale_sq * paid_m,
         pct_paid_im_ave_scale_sq_dollar = pct_paid_im_ave_scale_sq * (paid_i + paid_m)) %>%
  select( -slice, -paid_i, -paid_m)

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

#Parse ICD9
t1 <- claims %>%
  filter(!is.na(diagnosis_icd9_cd)) %>%
  select(claim_id, diagnosis_icd9_cd) %>%
  mutate(cat_icd1 = str_replace_all(diagnosis_icd9_cd, '[[:alpha:]]', ''),
         cat_icd2 = str_replace_all(diagnosis_icd9_cd, '\\.', '')) %>%
  mutate(cat_icd3 = ifelse(str_length(cat_icd1) > 3, str_sub(cat_icd1, 1, 3), cat_icd1))

t2 <- t1 %>%
  filter(str_detect(diagnosis_icd9_cd, '\\.')) %>%
  separate(diagnosis_icd9_cd, into=c('cat_icd4', 'cat_icd5'), remove=FALSE) %>%
  select(claim_id, cat_icd4:cat_icd5)

t3 <- t1 %>%
  select(claim_id, cat_icd1:cat_icd3) %>%
  left_join(t2) %>%
  mutate(cat_icd6 = coalesce(cat_icd4, cat_icd3))

#CAS CLM data
cas_clm <- claims %>%
  select(claim_id, cas_aia_cds_1_2, cas_aia_cds_3_4, clm_aia_cds_1_2, clm_aia_cds_3_4) %>%
  mutate(cat_clm1 = cas_aia_cds_1_2 %>% str_sub(1, 1),
         cat_clm2 = cas_aia_cds_1_2 %>% str_sub(2, 2),
         cat_clm3 = cas_aia_cds_3_4 %>% str_sub(1, 1),
         cat_clm4 = cas_aia_cds_3_4 %>% str_sub(2, 2),
         cat_clm5 = clm_aia_cds_1_2 %>% str_sub(1, 1),
         cat_clm6 = clm_aia_cds_1_2 %>% str_sub(2, 2),
         cat_clm7 = clm_aia_cds_3_4 %>% str_sub(1, 1),
         cat_clm8 = clm_aia_cds_3_4 %>% str_sub(2, 2)) %>%
  select(claim_id, contains('cat'))

#highest values values
train.3 <- train.data %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  select(claim_id, cutoff_year) %>%
  inner_join(claims %>%
               select(claim_id, one_of(cat.vars), one_of(cont.vars), one_of(date.vars))) %>%
  left_join(t3) %>%
  inner_join(cas_clm) %>%
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

         cat_1 = paste0(diagnosis_icd9_cd, state),
         cat_1a = paste0(cat_icd6, state),
         cat_2 = paste0(catas_or_jntcvg_cd, state),
         cat_3 = paste0(clm_aia_cds_1_2, clm_aia_cds_3_4),
         cat_4 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd),
         cat_4a = paste0(cat_icd6, catas_or_jntcvg_cd),

         cat_5 = paste0(clm_aia_cds_1_2, clm_aia_cds_3_4, state),
         cat_6 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd, state),
         cat_6a = paste0(cat_icd6, catas_or_jntcvg_cd, state),
         cat_7 = paste0(diagnosis_icd9_cd, clm_aia_cds_1_2, clm_aia_cds_3_4),
         cat_7a = paste0(cat_icd6, clm_aia_cds_1_2, clm_aia_cds_3_4),
         cat_8 = paste0(catas_or_jntcvg_cd, clm_aia_cds_1_2, clm_aia_cds_3_4),

         cat_9 = paste0(diagnosis_icd9_cd, state, clm_aia_cds_1_2, clm_aia_cds_3_4),
         cat_10 = paste0(state, catas_or_jntcvg_cd, clm_aia_cds_1_2, clm_aia_cds_3_4),
         cat_11 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd, clm_aia_cds_1_2, clm_aia_cds_3_4),

         cat_12 = paste0(diagnosis_icd9_cd, catas_or_jntcvg_cd, clm_aia_cds_1_2, clm_aia_cds_3_4, state)

         )


#checks
train.3 %>%
  filter(age_at_injury < 0)
train.3 %>%
  filter(loss_yearmo > cutoff_year)
train.3 %>%
  filter(empl_hire_yearmo < clmnt_birth_yearmo)

#Adj counts
train.4 <- train.data %>%
  group_by(claim_id) %>%
  count(adj) %>%
  spread(adj, n, fill = 0)

#Minor Codes
mc_codes <- train.data %>%
  group_by(claim_id) %>%
  select(mc_string) %>%
  filter(!is.na(mc_string))

mc_codes_long <- mc_codes %>%
#  mutate(mc_string = str_trim(mc_string),
#         mc_len1 = str_length(mc_string),
#         mc_len2 = str_count(mc_string, ' ')) %>%
separate(mc_string, into = paste0('code', 1:31), fill = 'right') %>%
  gather(variable, mc_code, - claim_id) %>%
  filter(!is.na(mc_code))

#Filter to codes in at least 1000 claims
mc_codes_cnt <- mc_codes_long %>%
  ungroup() %>%
  count(mc_code, sort = TRUE) %>%
  filter(n > 1000)

train.5 <- mc_codes_long %>%
  count(mc_code) %>%
  semi_join(mc_codes_cnt %>% select( - n)) %>%
  ungroup %>%
  full_join(claims %>%
              select(claim_id)) %>%
  complete(claim_id, mc_code, fill = list(n = 0)) %>%
  mutate(mc_code = paste0('mc_code_', mc_code)) %>%
  spread(mc_code, n, fill = 0)

train.data <- train.1 %>%
  inner_join(train.2) %>%
  inner_join(train.2d) %>%
  inner_join(train.3) %>%
  inner_join(train.4) %>%
  inner_join(train.5) %>%
  inner_join(reg.all %>%
              select( -paid_i, -paid_m, -slice))

cat.vars <- union(c("adj"), cat.vars)
new.cat.vars <- names(train.data) %>% str_subset('cat_')
cat.vars <- union(cat.vars, new.cat.vars)
cat.table.long <- NULL
for (i in cat.vars) {

  train.var <- train.data %>%
    group_by_(i) %>%
    summarise(n.train = n()) %>%
    mutate(pct.train = n.train / sum(n.train))

  var.table <- train.var %>%
    rename_(value = i) %>%
    mutate(var = i)

  cat.table.long <- bind_rows(cat.table.long, var.table)
}


cat.table.long <- cat.table.long %>%
  group_by(var) %>%
  mutate(pct.total = n.train / sum(n.train)) %>%
  arrange(var, desc(n.train)) %>%
  mutate(#start at 0
         cat.number = row_number() - 1,
         cat.number = ifelse(pct.train < .001, NA, cat.number)) %>%
  fill(cat.number)

x <- cat.table.long %>%
  group_by(var) %>%
  filter(cat.number==max(cat.number)) %>%
  distinct(var, cat.number)
sum(x$cat.number)

train.recode <- train.data %>% select(claim_id)
cat_count <- 1
for (i in cat.vars) {

  new_name <- paste0('cat_', cat_count)
  recode <- cat.table.long %>%
    ungroup() %>%
    filter(var == i) %>%
    select(value, cat.number) %>%
    setNames(c(i, 'cat.number'))

  pct.nm <- paste0(i, "_cat_pct")

  df_train <- train.data %>%
    select_('claim_id', i) %>%
    inner_join(recode) %>%
    #drop old variable
  select(-2) %>%
    setNames(c('claim_id', new_name))
  
  train.recode <- train.recode %>%
    inner_join(df_train)
  
  cat_count <- cat_count+1

}


train.data <- train.data %>%
  select(-one_of(cat.vars)) %>%
  inner_join(train.recode) %>%
  ungroup()

####Box Transform####
names(train.data)

train.data <- train.data %>%
  rename(cat_slice = slice)

cat.vars <- names(train.data) %>% str_subset("cat")
mc.vars <- names(train.data) %>% str_subset("mc_code")
names(train.data) %>% str_subset("sign")
binary.vars <- c('any_i', 'any_m', 'any_l', 'any_o', 'sign_paid_i', 'sign_paid_m', 'sign_paid_im')


#skew and unskewed versions of paids
skew.vars <- setdiff(names(train.data), cat.vars)
skew.vars <- setdiff(skew.vars, skew.vars %>% str_subset("any|paid|mean|hist|sign"))
skew.vars

cont.vars <- setdiff(names(train.data), cat.vars)
cont.vars <- setdiff(cont.vars, binary.vars)

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

train.file <- "data/train_recode_10.csv.gz"
write.csv(train.data, gzfile(train.file), row.names=FALSE)

test.file <- "data/test_recode_10.csv.gz"
write.csv(test.data, gzfile(test.file), row.names=FALSE)
