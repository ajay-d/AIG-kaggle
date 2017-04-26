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

train.stack <- filter(merge(filter(train,paid_year==2015),split),split==0) %>% arrange(claim_id)
buckets <- c(-Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 
             47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)
train.stack$target <- as.integer(cut(train.stack$target, buckets))
source('R/utils.r')
stack.act <- create_actual_matrix(train.stack$target,buckets)

# read 
model1.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/keras_10_models_02_hold0.238217.csv')
model2.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/keras_12_hold0.234477.csv')
model6.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/xgb_two_models_03_13_cnt_hold0.235513.csv')
model7.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/xgb_10_2_models_wshi_02_hold0.230944.csv')
model13.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/xgb.resample.slice.hold.csv')
model14.hold <- read.csv('C:/Users/wshi/Desktop/submission/hold/xgb.10.all.cnt.hold.csv')

model1 <- read.csv('C:/Users/wshi/Desktop/submission/test/keras_10_models_02_hold0.238217_all.csv')
model2 <- read.csv('C:/Users/wshi/Desktop/submission/test/keras_12_hold0.234477_all.csv')
model6 <- read.csv('C:/Users/wshi/Desktop/submission/test/xgb_two_models_03_13_cnt_hold0.235513_all.csv')
model7 <- read.csv('C:/Users/wshi/Desktop/submission/test/xgb_10_2_models_wshi_02_hold0.230944.csv')
model13 <- read.csv('C:/Users/wshi/Desktop/submission/test/xgb.resample.slice.csv')
model14 <- read.csv('C:/Users/wshi/Desktop/submission/test/xgb.10.all.cnt.csv')


fn <- function(wgt){
  wgt <- c(wgt,1-sum(wgt))
  p <- (wgt[1]*model1.hold[,-1]
        +wgt[2]*model2.hold[,-1]
        +wgt[3]*model6.hold[,-1]
        +wgt[4]*model7.hold[,-1]
        +wgt[5]*model13.hold[,-1]
        +wgt[6]*model14.hold[,-1]
  )
  
  lgls <- mlogloss(as.matrix(p), stack.act, buckets)
  return(lgls)
}

op <- optim(par=rep(1/6,5), fn,#lower=0,upper=1,
            method = "Nelder-Mead")
op <- op$par
op <- c(op,1-sum(op))

library(signal)
pred.test <- (op[1]*model1[,-1]
              +op[2]*model2[,-1]
              +op[3]*model6[,-1]
              +op[4]*model7[,-1]
              +op[5]*model13[,-1]
              +op[6]*model14[,-1]
)

a <- pred.test
a[,3:21] <- t(apply(a[,3:21],1,function(x)sgolayfilt(x,2)))
a[a>1- 10 ^ -15] <- 1 - 10 ^ -15
a[a<10 ^ -15] <- 10 ^ -15
a <- a/ rowSums(a)
a <- cbind(model13$claim_id,a)
colnames(a) <- colnames(ss)
write.csv(a,paste0('C:/Users/wshi/Desktop/stack.model.sgolay.csv'),row.names=FALSE)

