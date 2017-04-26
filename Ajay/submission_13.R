rm(list = ls(all = TRUE))

library(dplyr)
library(tidyr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read.csv("data/sample_submission.csv")

raja <- read.csv("submission/fin-sub-1-23.csv")
ajay <- read.csv("submission/blend_19Jan2017_21_54.csv.gz")
gbm.1 <- read.csv("submission/gbm_m4_data11.csv.gz") %>%
  setNames(names(sample))

gbm.2 <- read.csv("submission/gbm_m4_data12.csv.gz") %>%
  setNames(names(sample))

nn.1 <- bind_cols(sample %>% select(claim_id),
                   read.csv("submission/nn_1.csv.gz")) %>%
  setNames(names(sample))

nn.2 <- bind_cols(sample %>% select(claim_id),
                  read.csv("submission/nn_2.csv.gz")) %>%
  setNames(names(sample))


blend.1 <- raja %>%
  gather(variable, value, - claim_id) %>%
  bind_rows(ajay %>%
              gather(variable, value, - claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23204

blend.2 <- raja %>%
  gather(variable, value, - claim_id) %>%
  bind_rows(gbm.1 %>%
              gather(variable, value, - claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23158


blend.nn <- nn.1 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(nn.2 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.3 <- blend.nn %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23103

r1 <- raja %>%
  gather(variable, value, -claim_id)
a1 <- ajay %>%
  gather(variable, value, -claim_id)
g1 <- gbm.1 %>%
  gather(variable, value, -claim_id)
nn <- blend.nn %>%
  gather(variable, value, -claim_id)


cor(r1$value, a1$value)
cor(r1$value, g1$value)
cor(a1$value, g1$value)
cor(nn$value, r1$value)

cor(nn$value, g1$value)

blend.4 <- blend.nn %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.1 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id)) %>%  
  bind_rows(raja %>%
              gather(variable, value, -claim_id)) %>%    
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 4) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.4 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23119
