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
train.data <- read.csv("data/train_recode_5.csv.gz")
test.data <- read.csv("data/test_recode_5.csv.gz")

#######################GBM 1########################################################

set.seed(666)

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 21,
              "nthread" = 12, 
              "eta" = 0.01,
              
              "alpha" = 3,
              "colsample_bytree" = 1,
              "gamma" = 5,
              "lambda" = 3,
              "max_depth" = 15,
              "min_child_weight" = 5,
              "subsample" = 1)

#CV all data
model.y <- train.data %>%
  use_series(target)

model.data <- train.data %>%
  select(-claim_id, -target, -loss) %>%
  data.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
bst.cv <- xgb.cv(params = param, 
                 data = xgbtrain, 
                 nrounds = 5000, 
                 nfold = 5,
                 early_stopping_rounds = 5,
                 print_every_n = 50)

bst.cv$best_iteration
bst.cv$best_ntreelimit
bst.cv$evaluation_log[['test_mlogloss_mean']][bst.cv$best_iteration]
#0.2432572 @554 @ .01 eta
#public: .24030

round(bst.cv$best_iteration/.8)
bst.1 = xgb.train(params = param, 
                  data = xgbtrain, 
                  nrounds = round(bst.cv$best_iteration/.8),
                  print_every_n = 50)

# Compute feature importance matrix
importance_matrix_1 <- xgb.importance(colnames(xgbtrain), model = bst.1)

gbm.1 <- matrix(predict(bst.1, test.data %>% select(-claim_id) %>% data.matrix()), nrow=nrow(test.data), byrow=TRUE)
gbm.1 <- cbind(test.data$claim_id, gbm.1) %>%
  as.data.frame() %>%
  setNames(names(sample))
write.csv(gbm.1, gzfile("submission/data5_gbm_d15.csv.gz"), row.names=FALSE)