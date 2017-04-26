load('data/data.RData')
test.id <- select(test,claim_id)
train.id <- setdiff(select(train,claim_id),test.id)


in.train <- claims$claim_id%in%train.id$claim_id
loss.yr <- claims[in.train,] %>% select(claim_id,loss_yearmo)
test.slice <- claims[!in.train,] %>% select(claim_id,slice)

set.seed(123)
claim.id <- loss.yr$claim_id
slice.r <- matrix(NA,nrow=nrow(train.id),ncol=10)
for(i in 1:10){
  for(cut in seq(10,2,-1)){
    chosen <- loss.yr %>% 
      filter(loss_yearmo<2016-cut,is.na(slice.r[,i])) %>% 
      select(claim_id) %>% unlist() %>% sample(30838,replace=FALSE)
    slice.r[claim.id%in%chosen,i] <- cut
  }
  slice.r[is.na(slice.r[,i]),i] <- 1
}

slice.r <- cbind(claim.id,slice.r)
colnames(slice.r) <- c('claim_id','1','2','3','4','5','6','7','8','9','10')

a <- cbind(select(test.slice,claim_id),matrix(rep(test.slice$slice,10),ncol=10))
colnames(a) <- c('claim_id','1','2','3','4','5','6','7','8','9','10')

slice <- rbind(slice.r,a)

write.csv(slice,'data/slice.csv',row.names=FALSE)