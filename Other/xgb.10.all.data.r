
library(dplyr)
library(data.table)
library(sqldf)
library(moments)
library(xgboost)

rm(list = ls())

directory <- '/mnt/junwang/project6'
setwd(directory)

strt <- Sys.time()
slice <- dplyr::slice

load('data/data.RData')
slices <- select(claims,claim_id,slice)
colnames(slices) <- c('claim_id','slice')

train.2015 <- filter(train,paid_year==2015)
train.prior.2015 <- filter(train,paid_year<2015)

buckets <- c(-Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 
             47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)

# trim zeros
source('R/utils.r')
source('R/data.trim.zero.r')
train.prior.2015.trim <- data.trim.zero(train.prior.2015)


result.all <- list()
result.all.stack <- list()
for (s in 1:10){
  working.slice <- data.frame(claim_id=claims$claim_id,slice=s)
  # payment history
  source('R/data.payment.history.r')
  payment.history <- data.payment.history(train.prior.2015,working.slice)

  # payment summary
  source('R/data.payment.summary2.r')
  payment.summary <- data.payment.summary(train.prior.2015.trim,working.slice)

  # last mc to mat
  source('R/data.last.mc.mat.r')
  last.mc.mat <- data.last.mc.mat(payment.summary)

  # claims
  source('R/data.claims.r')
  train.2015 <- filter(train,paid_year==2015)
  split <- read.csv('data/split.csv')
  folds <- sort(unique(split$split))

  for (valid.fold in 0:0){
    train.fold <- setdiff(folds,c(valid.fold))
    train.fold.char <- paste0(train.fold,collapse='')
    working.claims <- data.claims(claims,train.2015,payment.summary,buckets,working.slice,split,train.fold)
  }
  
  
  # prepare dataset
  # turn 2015 into actuall target
  train.2015 <- filter(train,paid_year==2015)
  train.2015$target <- as.integer(cut(train.2015$target, buckets)) - 1
  
  # combind train test and cutoff year
  train.2015 <- merge(train.2015,working.slice)
  test <- merge(test,slices)
  train.test <- bind_rows(train.2015,test)
  train.test <- mutate(train.test,slice=ifelse(slice>10,slice-10,slice),
                       cutoff_year=2015-slice)
  train.test <- select(train.test,claim_id,cutoff_year,target,starts_with('econ_'))
  train.test <- merge(train.test,payment.history,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,payment.summary,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  train.test <- merge(train.test,last.mc.mat,by=c('claim_id','cutoff_year'),all.x=TRUE,all.y=FALSE)
  
  #
  source('R/create.dataset.all.r')
  split <- read.csv('data/split.csv')
  folds <- sort(unique(split$split))

  for (valid.fold in 0:0){
    train.fold <- setdiff(folds,c(valid.fold))
    train.fold.char <- paste0(train.fold,collapse='')
    working.train.test <- merge(train.test,working.claims,by='claim_id',all=TRUE)
    working.train.test <- filter(working.train.test,floor(loss_yearmo)<=cutoff_year)
    working.train.test <- filter(working.train.test,cutoff_year==2015-s)
    working.train.test <- create.dataset.all(working.train.test)
    write.csv(working.train.test,paste0('data/R_10_model_all/train.test.model.',s,'.all.cat.csv'),row.names=FALSE)
  }
}
