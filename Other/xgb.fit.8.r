
xgb.fit <- function(working.train.test,split,train.fold,valid.fold){

  library(dplyr)
  library(xgboost)
  packageVersion("xgboost")
  slice <- dplyr::slice
  
  working.train.test <- merge(working.train.test,split,all.x=TRUE,all.y=FALSE)

  working.train <- filter(working.train.test,!is.na(split)&split%in%train.fold) %>% arrange(claim_id)
  working.valid <- filter(working.train.test,!is.na(split),split%in%valid.fold) %>% arrange(claim_id)
  working.test <- filter(working.train.test,is.na(split)|(!split%in%c(train.fold,valid.fold))) %>% arrange(claim_id)
  
  # 
  X.train <- select(working.train,-claim_id,-target,-split)
  y.train <- working.train$target
  X.valid <- select(working.valid,-claim_id,-target,-split)
  y.valid <- working.valid$target
  X.test <- select(working.test,-claim_id,-target,-split)
  test.id <- select(working.test,claim_id)
  
  # covert to xgb dmatrix
  dtrain <- xgb.DMatrix(data.matrix(X.train),label=y.train,missing=NA)
  dvalid <- xgb.DMatrix(data.matrix(X.valid),label=y.valid,missing=NA)
  dtest <- xgb.DMatrix(data.matrix(X.test),missing=NA)
  
  set.seed(123)
  xgb_params <- list(
    seed=123,
    colsample_bytree=0.3,
    subsample=0.9,
    eta=0.1,
    lambda=0.2,
    alpha=0.8,
    gamma=0.4,
    max_depth=8,
    min_child_weight=20
  )
  
  model <- xgb.train(xgb_params,
                     dtrain,
                     watchlist=list(train=dtrain,valid=dvalid),
                     nthread=64,
                     nrounds=5000,
                     early_stopping_rounds=10,
                     print_every_n=10,
                     verbose=1,
                     objective='multi:softprob',
                     eval_metric = "mlogloss",
                     num_class = 21,
                     maximize=FALSE)
  
  # variable importance
  names <- names(X.train)
  importance_matrix <- xgb.importance(names, model = model)
  xgb.plot.importance(importance_matrix)
  
  pred_test <- matrix(predict(model, dtest), nrow = nrow(dtest), byrow = TRUE)
  
  result <- list()
  result$train.fold <- train.fold
  result$valid.fold <- valid.fold
  result$param <- xgb_params
  result$model <- model
  result$test.id <- test.id
  result$pred_test <- pred_test
  result$importance_matrix <- importance_matrix
  result$run_time <- Sys.time()
  return(result)

}