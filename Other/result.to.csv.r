
library(dplyr)
library(data.table)


rm(list = ls())

directory <- 'H:/Kaggle AIG'
setwd(directory)
load('data/data.RData')
slices <- select(claims,claim_id,slice)
split <- read.csv('data/split.csv')
train.slice <- merge(filter(train,paid_year==2015),select(claims,claim_id,slice))
train.slice$cutoff_year <- 2015-train.slice$slice

xgb.resample <- readRDS('results/xgb.resample.slice.0127.rds')
xgb.resample.stack <- readRDS('results/xgb.resample.slice.stack.0127.rds')
p <- NULL
for(i in 1:10){
  for(j in 1:4){
    if (is.null(p)){
      p <- xgb.resample[[i]][[j]]$pred_test
    }else{
      p <- p+xgb.resample[[i]][[j]]$pred_test
    }
  } 
}
p <- p/40
xgb.resample.pred.test <- cbind(xgb.resample[[1]][[1]]$test.id,p)

p <- NULL
for(i in 1:10){
  for(j in 1:3){
    if (is.null(p)){
      p <- xgb.resample.stack[[i]][[j]]$pred_test
    }else{
      p <- p+xgb.resample.stack[[i]][[j]]$pred_test
    }
  } 
}
p <- p/30
xgb.resample.stack.pred.stack.test <- cbind(xgb.resample.stack[[1]][[1]]$test.id,p)
xgb.resample.stack.pred.test <- merge(xgb.resample.stack.pred.stack.test,select(test,claim_id),all=FALSE)
xgb.resample.stack.pred.stack <- setdiff(xgb.resample.stack.pred.stack.test,xgb.resample.stack.pred.test)


xgb.10.all <- readRDS('results/xgb.10.model.all.cnt.0127.rds')
xgb.10.stack.all <- readRDS('results/xgb.10.model.stack.all.cnt.0127.rds')

xgb.10.all.pred.test <- NULL
for(i in 1:10){
  p <- NULL
  for(j in 1:4){
    if (is.null(p)){
      p <- xgb.10.all[[i]][[j]]$pred_test
    }else{
      p <- p+xgb.10.all[[i]][[j]]$pred_test
    }
  } 
  p <- p/4
  p <- cbind(xgb.10.all[[i]][[1]]$test.id,p)
  xgb.10.all.pred.test <- rbind(xgb.10.all.pred.test,p)
}


xgb.10.stack.all.pred.test <- NULL
xgb.10.stack.all.pred.stack <- NULL
train.slice <- merge(filter(train,paid_year==2015),select(claims,claim_id,slice))
train.slice$cutoff_year <- 2015-train.slice$slice
for(i in 1:10){
  p <- NULL
  for(j in 1:3){
    if (is.null(p)){
      p <- xgb.10.stack.all[[i]][[j]]$pred_test
    }else{
      p <- p+xgb.10.stack.all[[i]][[j]]$pred_test
    }
  } 
  p <- p/3
  xgb.10.stack.all.pred.stack.test <- cbind(xgb.10.stack.all[[i]][[1]]$test.id,p)
  
  tmp1 <- merge(xgb.10.stack.all.pred.stack.test,select(test,claim_id),all=FALSE)
  tmp2 <- merge(xgb.10.stack.all.pred.stack.test,select(filter(train.slice,cutoff_year==2015-i),claim_id),all=FALSE)
  
  xgb.10.stack.all.pred.test <- rbind(xgb.10.stack.all.pred.test,tmp1)
  xgb.10.stack.all.pred.stack <- rbind(xgb.10.stack.all.pred.stack,tmp2)
}


xgb.10.stack.all.pred.test<-arrange(xgb.10.stack.all.pred.test,claim_id)
xgb.10.stack.all.pred.stack<-arrange(xgb.10.stack.all.pred.stack,claim_id)
xgb.10.all.pred.test<-arrange(xgb.10.all.pred.test,claim_id)

xgb.resample.stack.pred.test<-arrange(xgb.resample.stack.pred.test,claim_id)
xgb.resample.stack.pred.stack<-arrange(xgb.resample.stack.pred.stack,claim_id)
xgb.resample.pred.test<-arrange(xgb.resample.pred.test,claim_id)


ss <- read.csv('data/sample_submission.csv')
colnames(xgb.10.pred.test) <- colnames(ss)
write.csv(xgb.resample.pred.test,paste0('C:/Users/wshi/Desktop/submission/test/xgb.resample.slice.csv'),row.names=FALSE)
colnames(xgb.10.all.pred.test) <- colnames(ss)
write.csv(xgb.10.all.pred.test,paste0('C:/Users/wshi/Desktop/submission/test/xgb.10.all.cnt.csv'),row.names=FALSE)

colnames(xgb.10.stack.pred.stack) <- colnames(ss)
write.csv(xgb.resample.stack.pred.stack,paste0('C:/Users/wshi/Desktop/submission/hold/xgb.resample.slice.hold.csv'),row.names=FALSE)
colnames(xgb.10.stack.all.pred.stack) <- colnames(ss)
write.csv(xgb.10.stack.all.pred.stack,paste0('C:/Users/wshi/Desktop/submission/hold/xgb.10.all.cnt.hold.csv'),row.names=FALSE)
