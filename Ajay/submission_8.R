rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read_csv("data/sample_submission.csv")

gbm.1 <- read_csv("submission/gbm_bag_m8a_d15.csv.gz") %>%
  setNames(names(sample))

gbm.2 <- read_csv("submission/gbm_m1_data8.csv.gz") %>%
  setNames(names(sample))

gbm.3 <- read_csv("submission/gbm_bag_m8b_d20.csv.gz") %>%
  setNames(names(sample))

gbm.4 <- read_csv("submission/gbm_m2_data8.csv.gz") %>%
  setNames(names(sample))

write.csv(gbm.1, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23704

write.csv(gbm.3, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23682

write.csv(gbm.4, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23618

blend.1 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
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

df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23679

blend.2 <- gbm.1 %>%
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

df <- blend.2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23683

blend.best <- read_csv('submission/blend_16Dec2016_20_54.csv.gz')
## .23634

blend.2 <- blend.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Blend 1') %>%
  bind_rows(blend.best %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='Blend 2')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23549

blend.3 <- gbm.4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 4') %>%
  bind_rows(blend.best %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='Blend Best')) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23494



