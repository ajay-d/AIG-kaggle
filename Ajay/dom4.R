##RUN eta 03 model
##use 7000 layers (dim)
##add cat features, 5k, 10k / adjust layers
##patience 5/10

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")

rm(list=ls(all=TRUE))

library(dplyr)
library(magrittr)
library(xgboost)

options(stringsAsFactors = FALSE,
        scipen = 10)

train.data <- read.csv("train_recode_5.csv.gz")
test.data <- read.csv("test_recode_5.csv.gz")
sample <- read.csv("sample_submission.csv")

set.seed(44)

parallel::detectCores()

param <- list("objective" = "multi:softprob",
              "eval_metric" = "mlogloss",
              "num_class" = 21,
              "nthread" = 32, 
              "eta" = 0.025,
              
              "alpha" = 4.87,
              "colsample_bytree" = .016,
              "gamma" = 12.67,
              "lambda" = 7.44,
              "max_depth" = 20,
              "min_child_weight" = 8.84,
              "subsample" = .727)

#CV all data
model.y <- train.data %>%
  use_series(target)

model.data <- train.data %>%
  select(-claim_id, -target, -loss) %>%
  data.matrix()

xgbtrain <- xgb.DMatrix(data=model.data, label=model.y)
bst.cv <- xgb.cv(params = param, 
                 data = xgbtrain, 
                 nrounds = 10000, 
                 nfold = 5,
                 early_stopping_rounds = 10,
                 print_every_n = 50)

bst.cv$best_iteration
bst.cv$best_ntreelimit
bst.cv$evaluation_log[['test_mlogloss_mean']][bst.cv$best_iteration]
#0.2401 @3041 @ .01 eta
#public: .

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
write.csv(gbm.1, gzfile("data5_dom4.csv.gz"), row.names=FALSE)