rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)

options(stringsAsFactors = FALSE,
        scipen = 10)

sample <- read_csv("data/sample_submission.csv")

blend.best <- read_csv('submission/blend_16Dec2016_20_54.csv.gz')
## .23634

gbm.m4 <- read_csv("submission/gbm_m4_data8.csv.gz") %>%
  setNames(names(sample))
## .23584

gbm.m5 <- read_csv("submission/gbm_m5_data8.csv.gz") %>%
  setNames(names(sample))
## .23584

gbm.m4_5 <- read_csv("submission/gbm_m4_data5.csv.gz") %>%
  setNames(names(sample))
## .23655

gbm.m2 <- read_csv("submission/gbm_m2_data8.csv.gz") %>%
  setNames(names(sample))

gbm.m6 <- read_csv("submission/gbm_m6_data8.csv.gz") %>%
  setNames(names(sample))

write.csv(gbm.m4_5, gzfile(paste0('submission/dom_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)


train.data <- read.csv("data/train_recode_8.csv.gz")
glimpse(train.data)
emp <- train.data %>%
  count(target) %>%
  mutate(pct = n/sum(n),
         target = target + 1) %>%
  as.data.frame()

gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  group_by(variable) %>%
  summarise(m = mean(value)) %>%
  separate(variable, c('Bucket', 'target'), convert=TRUE) %>%
  arrange(Num) %>%
  as.data.frame()

emp <- emp %>%
  inner_join(gbm.m4 %>%
              gather(variable, value, -claim_id) %>%
              group_by(variable) %>%
              summarise(m = min(value)) %>%
              separate(variable, c('Bucket', 'target'), convert=TRUE))

uniform.weights <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(value = 1/21)

gbm.m4.smooth <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(weighted.value = .99 * value) %>%
  bind_rows(uniform.weights %>%
            mutate(weighted.value = .01 * value)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(weighted.value)) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend) %>%
  select(one_of(names(sample)))

gbm.m4.smooth %>%
  gather(variable, value, -claim_id) %>%
  group_by(variable) %>%
  summarise(m = min(value)) %>%
  separate(variable, c('Bucket', 'target'), convert=TRUE)

write.csv(gbm.m4.smooth, gzfile(paste0('submission/gbm_m4_smooth_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .24335

gbm.m4.smooth <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(weighted.value = .999 * value) %>%
  bind_rows(uniform.weights %>%
            mutate(weighted.value = .001 * value)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(weighted.value)) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend) %>%
  select(one_of(names(sample)))

gbm.m4.smooth %>%
  gather(variable, value, -claim_id) %>%
  group_by(variable) %>%
  summarise(m = min(value)) %>%
  separate(variable, c('Bucket', 'target'), convert=TRUE)

write.csv(gbm.m4.smooth, gzfile(paste0('submission/gbm_m4_smooth_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23654

blend.1 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM 4') %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 5')) %>%
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
## .23582

blend.2 <- blend.best %>%
  gather(variable, value, -claim_id) %>%
  mutate(model='GBM best') %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id) %>%
              mutate(model='GBM 5')) %>%
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
## .23605

blend.1 <- blend.best %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m4 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.2 <- gbm.m5 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

blend.1_2 <- blend.1 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(blend.2 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/2) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.1_2 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23432

blend.3 <- gbm.m4 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m4_5 %>%
              gather(variable, value, -claim_id)) %>%
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
## .23422

#####################################################################

blend.2_4_5_6 <- gbm.m2 %>%
  gather(variable, value, -claim_id) %>%
  bind_rows(gbm.m4 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m5 %>%
              gather(variable, value, -claim_id)) %>%
  bind_rows(gbm.m6 %>%
              gather(variable, value, -claim_id)) %>%
  group_by(claim_id, variable) %>%
  summarise(value = sum(value)) %>%
  mutate(value = value/4) %>%
  group_by(claim_id) %>%
  mutate(value_blend = value/sum(value)) %>%
  select(claim_id, variable, value_blend) %>%
  spread(variable, value_blend)

df <- blend.2_4_5_6 %>%
  select(one_of(names(sample)))
write.csv(df, gzfile(paste0('submission/blend_', format(Sys.time(), "%d%b%Y_%H_%M"), '.csv.gz')), row.names=FALSE)
## .23589
