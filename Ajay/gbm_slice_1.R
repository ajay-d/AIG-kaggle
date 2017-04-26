rm(list = ls(all = TRUE))

library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
library(stringr)
library(xgboost)
library(magrittr)

options(stringsAsFactors = FALSE,
        scipen = 10)

train.data <- read.csv("data/train_recode_9.csv.gz")
test.data <- read.csv("data/test_recode_9.csv.gz")
sample <- read_csv("data/sample_submission.csv")

set.seed(777)

importance_matrix <- NULL
test_scored <- NULL
param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 21,
              "nthread" = 12,
              "eta" = 0.02,
              "silent" = 0,
              "alpha" = 3,
              "colsample_bytree" = .25,
              "gamma" = 5,
              "lambda" = 3,
              "max_depth" = 20,
              "min_child_weight" = 5,
              "subsample" = 1)

for (i in 1:2) {
  
  print(paste0("Slice ", i))

  model.y <- train.data %>%
    filter(slice == i) %>%
    use_series(target)

  model.data <- train.data %>%
    filter(slice == i) %>%
    select(-claim_id, -target, -loss) %>%
    data.matrix()

  xgbtrain <- xgb.DMatrix(data = model.data, label = model.y)

  bst.cv <- xgb.cv(params = param,
                 data = xgbtrain,
                 #nrounds = 5000,
                 nrounds = 100,
                 nfold = 5,
                 early_stopping_rounds = 5,
                 print_every_n = 50)

  round(bst.cv$best_iteration / .8)
  bst.1 = xgb.train(params = param,
                    data = xgbtrain,
                    nrounds = round(bst.cv$best_iteration / .8),
                    print_every_n = 50)

  # Compute feature importance matrix
  importance_matrix_1 <- xgb.importance(colnames(xgbtrain), model = bst.1) %>%
    mutate(logloss = bst.cv$evaluation_log[['test_mlogloss_mean']][bst.cv$best_iteration],
           iternations = bst.cv$best_iteration,
           slice = i)

  importance_matrix <- bind_rows(importance_matrix, importance_matrix_1)

  test_i <- test.data %>%
    filter(slice == i)

  gbm.1 <- matrix(predict(bst.1, test_i %>% select(-claim_id) %>% data.matrix()), 
                  nrow = nrow(test_i), byrow = TRUE)

  gbm.1 <- cbind(test_i$claim_id, gbm.1) %>%
    as.data.frame() %>%
    setNames(names(sample))

  test_scored <- bind_rows(test_scored, gbm.1)
}

write.csv(test_scored, gzfile("submission/gbm_by_slice1.csv.gz"), row.names = FALSE)
write.csv(importance_matrix, "importance_by_slice1.csv", row.names = FALSE)