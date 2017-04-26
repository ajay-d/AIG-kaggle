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

gbm.1 <- read_csv("submission/gbm_bag_m2a_d15.csv") %>%
  setNames(names(sample))

gbm.2 <- read_csv("submission/gbm_bag_m2_d15_norm.csv.gz") %>%
  setNames(names(sample))

gbm.3 <- read_csv("submission/gbm_bag_m1a_d15.csv.gz") %>%
  setNames(names(sample))

gbm.4 <- read_csv("submission/gbm_bag_m3b_d10.csv.gz") %>%
  setNames(names(sample))

gbm.5 <- read_csv("submission/gbm_bag_m4a_d20.csv.gz") %>%
  setNames(names(sample))

write.csv(gbm.5, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23776

blend.1 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 2')) %>%
  bind_rows(gbm.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23634

blend.2 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(gbm.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%
  bind_rows(gbm.4 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 4')) %>%  
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
## .23649

blend.3 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(gbm.3 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 3')) %>%
  bind_rows(gbm.5 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 5')) %>%  
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
## .23653