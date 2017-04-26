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

#cv2 @ .03
gbm.2 <- read_csv("submission/gbm_slice2_cv1.csv.gz")

#cv3 @ .03
gbm.3 <- read_csv("submission/gbm_slice3_cv1.csv.gz")


#0.2437416 @495 @ .03 eta
#public: .24101
gbm.2

#0.2432572 @554 @ .03 eta
#public: .24030
gbm.3

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
## .24050

set.seed(777)

#######################GBM 2 - deeper########################################################

train.data <- train.data %>%
  inner_join(train.stack)

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
                 early_stopping_rounds = 25,
                 print_every_n = 50)

bst.cv$best_iteration
bst.cv$best_ntreelimit
bst.cv$evaluation_log[['test_mlogloss_mean']][bst.cv$best_iteration]
# @

round(bst.cv$best_iteration/.8)
bst.2 = xgb.train(params = param, 
                  data = xgbtrain, 
                  nrounds = round(bst.cv$best_iteration/.8),
                  print_every_n = 50)

# Compute feature importance matrix
importance_matrix_2 <- xgb.importance(colnames(xgbtrain), model = bst.2)

test.data <- test.data %>%
  inner_join(test.stack)

gbm.2 <- matrix(predict(bst.2, test.data %>% select(-claim_id) %>% data.matrix()), nrow=nrow(test.data), byrow=TRUE)
gbm.2 <- cbind(test.data$claim_id, gbm.2) %>%
  as.data.frame() %>%
  setNames(names(sample))
write.csv(gbm.2, gzfile("submission/gbm_m1_stack_1.csv.gz"), row.names=FALSE)
#.23869

