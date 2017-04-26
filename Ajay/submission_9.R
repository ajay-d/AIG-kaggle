rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read_csv("data/sample_submission.csv")

blend.best <- read_csv('submission/blend_16Dec2016_20_54.csv.gz')
## .23634

gbm.m2 <- read_csv("submission/gbm_m2_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m4 <- read_csv("submission/gbm_m4_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m5 <- read_csv("submission/gbm_m5_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m6 <- read_csv("submission/gbm_m6_data8.csv.gz") %>%
  setNames(names(sample))

write.csv(gbm.m2, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23618

write.csv(gbm.m4, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23584

write.csv(gbm.m6, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23611

blend.1 <- gbm.m2 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 2') %>%
  bind_rows(gbm.m6 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 6')) %>%
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
## ??
## .23611

blend.2 <- blend.best %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Blend best') %>%
  bind_rows(blend.1 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='Blend 1')) %>%
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
## .23489

blend.3 <- blend.best %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='Blend best') %>%
  bind_rows(gbm.m4 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 4')) %>%
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
## .23470
