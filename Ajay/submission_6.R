rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)
library(magrittr)
library(xgboost)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read_csv("data/sample_submission.csv")

gbm.1 <- read_csv("submission/gbm_m1_stack_1.csv.gz")

gbm.2 <- read_csv("submission/gbm_m2_eta01_data3.csv") %>%
  setNames(names(sample))

gbm.3 <- read_csv("submission/gbm_slice3_cv1.csv.gz")
gbm.4 <- read_csv("submission/gbm_m1_f5_eta01_data4.csv") %>%
  setNames(names(sample))

dom.1 <- read_csv("submission/data5_dom1.csv.gz", col_names=FALSE) %>%
  setNames(names(sample))

dom.3 <- read_csv("submission/data5_dom3_eta_03.csv.gz", col_names=FALSE) %>%
  setNames(names(sample))

df <- dom.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23792

df <- dom.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23778

blend.1 <- gbm.2 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 2') %>%
  bind_rows(gbm.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)
## .23879

blend.2 <- blend.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Model 1') %>%
  bind_rows(dom.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='Model 2')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)


df <- blend.2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23649


gbm.4
## .23839

blend.3 <- gbm.4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 4') %>%
  bind_rows(dom.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 1')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)


df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23721

blend.4 <- dom.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(dom.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.4 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23708

blend.5 <- blend.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Blend 1') %>%
  bind_rows(dom.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%
  bind_rows(dom.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 1')) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.5 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23615

