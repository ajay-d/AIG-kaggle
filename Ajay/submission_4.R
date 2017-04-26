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
train.data <- read_csv("data/train_recode_3.csv.gz")
test.data <- read_csv("data/test_recode_3.csv.gz")

train.stack <- read_csv("data/train_stack.csv.gz")
test.stack <- read_csv("data/test_stack.csv.gz")

#######################################################

train.data <- train.data %>%
  inner_join(train.stack %>%
               setNames(c('claim_id', paste0('stack_', 1:(length(train.stack)-1)))))

test.data <- test.data %>%
  inner_join(test.stack %>%
               setNames(c('claim_id', paste0('stack_', 1:(length(test.stack)-1)))))

train.file <- "data/train_recode_4.csv.gz"
write.csv(train.data, gzfile(train.file), row.names=FALSE)

test.file <- "data/test_recode_4.csv.gz"
write.csv(test.data, gzfile(test.file), row.names=FALSE)

#######################################################

gbm_m1_stack_2 <- read_csv("submission/gbm_m1_stack_2.csv.gz")
##.47237

gbm.1 <- read_csv("submission/gbm_m1_eta01_data3.csv") %>%
  setNames(names(sample))

write.csv(gbm.1, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .24045

gbm.2 <- read_csv("submission/gbm_m2_eta01_data3.csv") %>%
  setNames(names(sample))

write.csv(gbm.2, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23992

#cv3 @ .03
gbm.3 <- read_csv("submission/gbm_slice3_cv1.csv.gz")

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


df <- blend.1 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23879

blend.2 <- gbm.1 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 1') %>%
  bind_rows(gbm.2 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 2')) %>%
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
## .23882

blend.3 <- gbm.1 %>%
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


df <- blend.3 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/ajay_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23901
