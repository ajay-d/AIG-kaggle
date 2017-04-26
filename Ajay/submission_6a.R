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

gbm.1 <- read_csv("submission/gbm_bag_m2_d20.csv.gz") %>%
  setNames(names(sample))

gbm.2 <- read_csv("submission/gbm_bag_m2a_d15.csv") %>%
  setNames(names(sample))

gbm.3 <- read_csv("submission/gbm_bag_m2_d15.csv.gz") %>%
  setNames(names(sample))

local.1 <- read_csv("submission/gbm_m2_eta01_data5.csv.gz") %>%
  setNames(names(sample))

old.1 <- read_csv("submission/gbm_m2_eta01_data3.csv") %>%
  setNames(names(sample))
old.2 <- read_csv("submission/gbm_slice3_cv1.csv.gz")

write.csv(gbm.1, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23744

write.csv(gbm.2, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23739

blend.1 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Model 1') %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='Model 2')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)


df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23729

blend.1 <- old.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(old.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 1')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.2 <- blend.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Blend 1') %>%
  bind_rows(gbm.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 1')) %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 2')) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23628


blend.3 <- local.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='local 1') %>%
  bind_rows(gbm.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 1')) %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 2')) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23727
