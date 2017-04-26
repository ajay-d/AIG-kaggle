# install.packages("drat", repos="https://cran.rstudio.com")
# drat:::addRepo("dmlc")
# install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")

library(dplyr)
library(data.table)
library(sqldf)
library(moments)
library(xgboost)

rm(list = ls())

directory <- 'H:/Kaggle AIG'
setwd(directory)

strt <- Sys.time()
slice <- dplyr::slice

load('data/data.RData')
slices <- read.csv('data/slice.csv')

train.2015 <- filter(train,paid_year==2015)
train.prior.2015 <- filter(train,paid_year<2015)

buckets <- c(-Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 
             47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)

source('R/utils.r')


result.all <- list()
result.all.stack <- list()
for (s in 1:10){
  working.slice <- data.frame(claim_id=slices$claim_id,slice=slices[,s+1])
  # payment history
  source('R/data.payment.history.r')
  payment.history <- data.payment.history(train.prior.2015,working.slice)
  
  # payment summary
  source('R/data.payment.summary.r')
  payment.summary <- data.payment.summary(train.prior.2015,working.slice)
  
  # last mc to mat
  source('R/data.last.mc.mat.r')
  last.mc.mat <- data.last.mc.mat(payment.summary)
  
  # claims
  source('R/data.claims.r')
  source('R/create.dataset.r')
  source('R/xgb.fit.r')
  split <- read.csv('data/split.csv')
  folds <- sort(unique(split$split))
  train.2015 <- filter(train,paid_year==2015)
  train.2015$target <- as.integer(cut(train.2015$target, buckets)) - 1
  
  # combind train test and cutoff year
  train.2015 <- merge(train.2015,working.slice)
  test <- merge(test,working.slice)
  train.test <- bind_rows(train.2015,test)
  train.test <- mutate(train.test,slice=ifelse(slice>10,slice-10,slice),
                       cutoff_year=2015-slice)
  train.test <- select(train.test,claim_id,cutoff_year,target,starts_with('econ_'))
  train.test <- merge(train.test,payment.history,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,payment.summary,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,last.mc.mat,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  
  result <- NULL
  for (valid.fold in 0:3){
    train.fold <- setdiff(folds,c(valid.fold))
    train.fold.char <- paste0(train.fold,collapse='')
    working.claims <- data.claims(claims,filter(train,paid_year==2015),payment.summary,buckets,slices,split,train.fold)
    working.train.test <- merge(train.test,working.claims,by='claim_id',all=TRUE)
    working.train.test <- filter(working.train.test,floor(loss_yearmo)<=cutoff_year)
    working.train.test <- create.dataset(working.train.test)
    xgb.result <- xgb.fit(working.train.test,split,train.fold,valid.fold)
    result[[valid.fold+1]] <- xgb.result
  }
  result.all[[s]] <- result
}

saveRDS(result.all,'results/xgb.resample.slice.cnt.0127.rds')


split <- read.csv('data/split.csv')
folds <- sort(unique(split$split))
slices <- read.csv('data/slice.csv')
slices <- merge(slices,select(claims,claim_id,slice),all=TRUE)
slices <- merge(slices,split,all=TRUE)
for (s in 1:10){
  slices[!is.na(slices$split)&slices$split==0,s+1] <- slices$slice[!is.na(slices$split)&slices$split==0]
}

slices <- select(slices,-split,-slice)
result.all.stack <- list()
for (s in 1:10){
  working.slice <- data.frame(claim_id=slices$claim_id,slice=slices[,s+1])
  
  # payment history
  source('R/data.payment.history.r')
  payment.history <- data.payment.history(train.prior.2015,working.slice)
  
  # payment summary
  source('R/data.payment.summary.r')
  payment.summary <- data.payment.summary(train.prior.2015,working.slice)
  
  # last mc to mat
  source('R/data.last.mc.mat.r')
  last.mc.mat <- data.last.mc.mat(payment.summary)
  
  source('R/data.claims.r')
  source('R/create.dataset.r')
  source('R/xgb.fit.r')
  split <- read.csv('data/split.csv')
  folds <- sort(unique(split$split))
  train.2015 <- filter(train,paid_year==2015)
  train.2015$target <- as.integer(cut(train.2015$target, buckets)) - 1
  train.2015 <- merge(train.2015,working.slice)
  test <- merge(test,working.slice)
  train.test <- bind_rows(train.2015,test)
  train.test <- mutate(train.test,slice=ifelse(slice>10,slice-10,slice),
                       cutoff_year=2015-slice)
  train.test <- select(train.test,claim_id,cutoff_year,target,starts_with('econ_'))
  train.test <- merge(train.test,payment.history,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,payment.summary,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,last.mc.mat,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  
  result <- NULL
  for (valid.fold in 1:3){
    train.fold <- setdiff(folds,c(0,valid.fold))
    train.fold.char <- paste0(train.fold,collapse='')
    working.claims <- data.claims(claims,filter(train,paid_year==2015),payment.summary,buckets,slices,split,train.fold)
    working.train.test <- merge(train.test,working.claims,by='claim_id',all=TRUE)
    working.train.test <- filter(working.train.test,floor(loss_yearmo)<=cutoff_year)
    working.train.test <- create.dataset(working.train.test)
    xgb.result <- xgb.fit(working.train.test,split,train.fold,valid.fold)
    result[[valid.fold]] <- xgb.result
  }
  result.all.stack[[s]] <- result
}

saveRDS(result.all.stack,'results/xgb.resample.slice.stack.cnt.0127.rds')


