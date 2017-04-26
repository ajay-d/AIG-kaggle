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

train.stack <- read_csv("data/train_stack2.csv.gz")
test.stack <- read_csv("data/test_stack2.csv.gz")

train.knn.stack <- read_csv("data/train_knn_stack.csv.gz")
test.knn.stack <- read_csv("data/test_knn_stack.csv.gz")

#######################################################

train.data <- train.data %>%
  inner_join(train.stack %>%
               setNames(c('claim_id', paste0('stack_', 1:(length(train.stack)-1))))) %>%
  inner_join(train.knn.stack %>%
             setNames(c('claim_id', paste0('stack_knn_', 1:(length(train.knn.stack)-1)))))

test.data <- test.data %>%
  inner_join(test.stack %>%
               setNames(c('claim_id', paste0('stack_', 1:(length(test.stack)-1))))) %>%
  inner_join(test.knn.stack %>%
               setNames(c('claim_id', paste0('stack_knn_', 1:(length(test.knn.stack)-1)))))

train.labels <- train.data %>%
  select(claim_id, loss, target)

train.data <- train.data %>%
  select(-loss, -target)

data.all <- bind_rows(train.data, test.data)

stack.vars <- data.all %>% 
  select(contains('stack')) %>%
  names

for (i in 2:length(data.all)) {
  nm <- names(data.all)[i]

  ##Scale numeric data
  if(nm %in% stack.vars)
    data.all[[nm]] <- scale(data.all[[nm]])
}


####Prepare test data####
test.data.norm <- data.all %>%
  anti_join(train.labels)

train.data.norm <- data.all %>%
  inner_join(train.labels)

train.file <- "data/train_recode_5.csv.gz"
write.csv(train.data.norm, gzfile(train.file), row.names=FALSE)

test.file <- "data/test_recode_5.csv.gz"
write.csv(test.data.norm, gzfile(test.file), row.names=FALSE)

#######################################################