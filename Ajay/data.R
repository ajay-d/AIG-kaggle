rm(list=ls(all=TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

claims <- read_csv("data/claims.csv")
train <- read_csv("data/train.csv")
test <- read_csv("data/test.csv")
sample <- read_csv("data/sample_submission.csv")

train %>% count(claim_id, sort=TRUE)
nrow(test)/nrow(train)
test %>% count(claim_id, sort=TRUE)
test %>% count(paid_year)

claims.ids <- sort(unique(claims$claim_id))
train.ids <- sort(unique(train$claim_id))
test.ids <- sort(unique(test$claim_id))
intersect(test.ids, train.ids) %>% length()
intersect(test.ids, claims.ids) %>% length()

train %>%
  count(claim_id, paid_year) %>%
  filter(n > 1)

train.2015.ids <- train %>%
  filter(paid_year==2015) %>%
  use_series(claim_id) %>%
  unique

train.2015 <- train %>%
  filter(claim_id %in% train.2015.ids)
length(train.2015.ids) + length(test.ids)

##42 states
test %>% count(econ_unemployment_py)

ay <- train %>%
  select(claim_id, paid_year) %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year))

#last paid years
ggplot(ay, aes(x=paid_year)) +
  geom_histogram()
#last paid years
ggplot(ay %>%
         filter(paid_year < 2015), aes(x=paid_year)) +
  geom_histogram()

ay %>%
  ungroup() %>%
  count(paid_year)

#in the test set, what's the last year we have
ay <- train %>%
  select(claim_id, paid_year) %>%
  filter(claim_id %in% test.ids) %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year))

#last year of the test set
ay %>%
  ungroup() %>%
  count(paid_year)

ggplot(ay, aes(x=paid_year)) +
  geom_histogram()

#in the train set, what's the last year we have
ay <- train %>%
  select(claim_id, paid_year) %>%
  filter(claim_id %in% train.2015.ids) %>%
  filter(paid_year < 2015) %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year))

#last year of the train set
ay %>%
  ungroup() %>%
  count(paid_year)

#year history for those in training set
ay <- train %>%
  select(claim_id, paid_year) %>%
  filter(claim_id %in% train.2015.ids)

ay %>% count(paid_year, sort=TRUE)

ggplot(ay, aes(x=paid_year)) +
  geom_histogram(binwidth=1)

ggplot(claims, aes(x=slice)) +
  geom_histogram(binwidth=.5)

train %>% count(adj)
train %>% 
  filter(paid_year == 2015) %>%
  count(econ_unemployment_py)
test %>% count(econ_unemployment_py)

################
test %>% 
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  count(slice)
train %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  count(slice)

train %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  anti_join(test %>%
              select(claim_id)) %>%
  count(slice)

test %>% 
  select(claim_id) %>%
  inner_join(train) %>%
  group_by(claim_id) %>%
  filter(paid_year == max(paid_year)) %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  ungroup %>%
  count(slice, paid_year)

train %>% 
  group_by(claim_id) %>%
  filter(paid_year == min(paid_year)) %>%
  inner_join(claims %>%
               select(claim_id, slice)) %>%
  #train data only
  filter(slice <= 10) %>%
  select(claim_id, paid_year, slice)
