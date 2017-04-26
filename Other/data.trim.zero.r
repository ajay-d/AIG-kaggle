data.trim.zero <- function(train.prior.2015){
  library(dplyr)
  library(data.table)
  slice <- dplyr::slice
  
  # remove consecutive 0s prior to 2015
  train.prior.2015.trim <- train.prior.2015
  flag <- TRUE
  while(flag){
    last.year <- train.prior.2015.trim %>% group_by(claim_id) %>% summarise(last_year=max(paid_year))
    train.prior.2015.trim <- merge(train.prior.2015.trim,last.year,all=TRUE)
    train.prior.2015.trim$flag <- 0
    train.prior.2015.trim$flag[(train.prior.2015.trim$paid_year==train.prior.2015.trim$last_year)&(train.prior.2015.trim$target==0)] <- 1
    if (sum(train.prior.2015.trim$flag)>0){
      flag = TRUE
    }else{
      flag = FALSE
    }
    train.prior.2015.trim <- filter(train.prior.2015.trim,flag<1)
    train.prior.2015.trim <- select(train.prior.2015.trim,-last_year,-flag)
  }
  
  # also trim from beginning
  flag <- TRUE
  while(flag){
    first.year <- train.prior.2015.trim %>% group_by(claim_id) %>% summarise(first_year=min(paid_year))
    train.prior.2015.trim <- merge(train.prior.2015.trim,first.year,all=TRUE)
    train.prior.2015.trim$flag <- 0
    train.prior.2015.trim$flag[(train.prior.2015.trim$paid_year==train.prior.2015.trim$first_year)&(train.prior.2015.trim$target==0)] <- 1
    if (sum(train.prior.2015.trim$flag)>0){
      flag = TRUE
    }else{
      flag = FALSE
    }
    train.prior.2015.trim <- filter(train.prior.2015.trim,flag<1)
    train.prior.2015.trim <- select(train.prior.2015.trim,-first_year,-flag)
  }
  
  return(train.prior.2015.trim)
}
