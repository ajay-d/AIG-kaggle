rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read_csv("data/sample_submission.csv")

gbm.m4 <- read_csv("submission/gbm_m4_data8.csv.gz") %>%
  setNames(names(sample))
## .23584

gbm.m5 <- read_csv("submission/gbm_m5_data8.csv.gz") %>%
  setNames(names(sample))
## .23584

gbm.m4_5 <- read_csv("submission/gbm_m4_data5.csv.gz") %>%
  setNames(names(sample))
## .23655

gbm.m4_5_norm <- read_csv("submission/gbm_m4_data5_norm.csv.gz") %>%
  setNames(names(sample))

gbm.m4_3 <- read_csv("submission/gbm_m4_data3.csv.gz") %>%
  setNames(names(sample))

gbm.m8a <- read_csv("submission/gbm_m8a_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m8b <- read_csv("submission/gbm_m8b_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m8f6 <- read_csv("submission/gbm_m4_data8_f6.csv.gz") %>%
  setNames(names(sample))

blend.1 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/4) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23413

blend.1 <- gbm.m4_3 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.2 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(blend.1 %>%
              gather(variable, value, -claim_id)) %>%  
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
## .23456

blend.3 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m8b %>%
              gather(variable, value, -claim_id)) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/5) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23405

blend.4 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m8b %>%
              gather(variable, value, -claim_id)) %>%  
  bind_rows(gbm.m8f6 %>%
              gather(variable, value, -claim_id)) %>%    
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/6) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.4 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23414

blend.5 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m8a %>%
              gather(variable, value, -claim_id)) %>%  
  bind_rows(gbm.m8b %>%
              gather(variable, value, -claim_id)) %>%    
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/6) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.5 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23414

blend.a <- gbm.m8a %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m8b %>%
              gather(variable, value, -claim_id)) %>%    
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.6 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(blend.a %>%
              gather(variable, value, -claim_id)) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/5) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.6 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23405

