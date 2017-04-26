rm(list = ls(all = TRUE))

library(dplyr)
library(tidyr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read.csv("data/sample_submission.csv")

gbm.m4_d8 <- read.csv("submission/gbm_m4_data8.csv.gz") %>%
  setNames(names(sample))
## .23584

gbm.m4a_d9 <- read.csv("submission/gbm_m4a_data9.csv.gz") %>%
  setNames(names(sample))
## .23566

write.csv(gbm.m4a_d9, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)

blend.1 <- gbm.m4_d8 %>%
  gather(variable, value, - claim_id) %>%
  bind_rows(gbm.m4a_d9 %>%
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
## .23521

gbm.m5 <- read.csv("submission/gbm_m5_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m4_5 <- read.csv("submission/gbm_m4_data5.csv.gz") %>%
  setNames(names(sample))

gbm.m4_5_norm <- read.csv("submission/gbm_m4_data5_norm.csv.gz") %>%
  setNames(names(sample))

gbm.m8b <- read.csv("submission/gbm_m8b_data8.csv.gz") %>%
  setNames(names(sample))

blend.3 <- gbm.m4a_d9 %>%
  gather(variable, value, - claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, - claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, - claim_id)) %>%
  bind_rows(gbm.m4_5_norm %>%
              gather(variable, value, - claim_id)) %>%
  bind_rows(gbm.m8b %>%
              gather(variable, value, - claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 5) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23405