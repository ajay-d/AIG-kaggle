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

nn.stack <- read.csv("data/lgblvl1_nnlvl2_data12_1.csv.gz") %>%
  setNames(names(sample))
# .24125

nn.3 <- read.csv("submission/merged_nn_top50cats.csv")


blend.nn <- nn.1 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(nn.2 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(nn.stack %>%
              gather(variable, value, -claim_id)) %>%  
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value / 3) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value / sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)


df <- nn.stack %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/nn_stack_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
# .24125

blend.1 <- blend.nn %>%
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

df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names = FALSE)
## .23085
