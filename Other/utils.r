
#2 way count 
my.f2cnt<-function(th2, vn1, vn2, filter=TRUE) { 
  df<-data.frame(f1=th2[,vn1], f2=th2[,vn2], filter=filter) 
  sum1<-sqldf("select f1, f2, count(*) as cnt from df where filter=1 group by 1,2") 
  tmp<-sqldf("select b.cnt from df a left join sum1 b on a.f1=b.f1 and a.f2=b.f2") 
  tmp$cnt[is.na(tmp$cnt)]<-0 
  return(tmp$cnt) 
} 


#3 way count 
my.f3cnt<-function(th2, vn1, vn2, vn3, filter=TRUE) { 
  df<-data.frame(f1=th2[,vn1], f2=th2[,vn2], f3=th2[, vn3], filter=filter) 
  sum1<-sqldf("select f1, f2, f3, count(*) as cnt from df where filter=1 group by 1,2, 3") 
  tmp<-sqldf("select b.cnt from df a left join sum1 b on a.f1=b.f1 and a.f2=b.f2 and a.f3=b.f3") 
  tmp$cnt[is.na(tmp$cnt)]<-0 
  return(tmp$cnt) 
} 

#shrank and randomized leave-one-out average actual for categorical variables 
my_exp1<-function(d1, vn1, vn2, y, vnp, filter, cred_k, r_k=.3){ 
  d2<-d1[, c(vn1, vn2, y, vnp)] 
  names(d2)<-c("f1", "f2", "a", "p") 
  d2$filter<-filter 
  d2$f1[is.na(d2$f1)] <- ''
  d2$f2[is.na(d2$f2)] <- ''
  sum1<-sqldf("select f1, f2, sum(1) as cnt, sum(p) as sump, sum(a) as suma from d2 where filter=1 group by 1,2") 
  tmp1<-sqldf("select a.a,a.p, b.cnt, b.sump, b.suma from d2 a left join sum1 b on a.f1=b.f1 and a.f2=b.f2") 
  tmp1$cnt[is.na(tmp1$cnt)]<-0 
  tmp1$avgp<-with(tmp1, sump/cnt) 
  tmp1$avgp[is.na(tmp1$avgp)]<-0 
  tmp1$suma[is.na(tmp1$suma)]<-0 
  tmp1$cnt[filter]<-tmp1$cnt[filter]-1 
  tmp1$suma[filter]<-tmp1$suma[filter]-tmp1$a[filter]
  tmp1$exp_a<-with(tmp1, suma/cnt) 
  tmp1$adj_a<-with(tmp1, (suma+p*cred_k)/(cnt+cred_k)) 
  tmp1$exp_a[is.na(tmp1$exp_a)]<-tmp1$p[is.na(tmp1$exp_a)] 
  tmp1$adj_a[is.na(tmp1$adj_a)]<-tmp1$p[is.na(tmp1$adj_a)] 
  tmp1$adj_a[filter]<-tmp1$adj_a[filter]*(1+(runif(sum(filter))-0.5)*r_k)
  
  if(vn1=='dummy'){
    flag <- (d2$f2=='')
  }else{
    flag <- (d2$f1=='')&(d2$f2=='')
  }
  tmp1$adj_a[flag] <- NA
  
  return(tmp1) 
} 

create_actual_matrix <- function(actual_vector, cut_points) {
  actual_m <- matrix(0, ncol = 21, nrow = length(actual_vector))
  actual_bucket <- as.integer(actual_vector)
  for (i in 1:nrow(actual_m)) actual_m[i, actual_bucket[i]] <- 1
  return(actual_m)
}

my_cat1<-function(d1, vn1, vn2, y, vnp, filter, cred_k, r_k=.3){ 
  d2<-d1[, c(vn1, vn2, y, vnp)] 
  names(d2)<-c("f1", "f2", "a", "p") 
  d2$filter<-filter 
  d2$f1[is.na(d2$f1)] <- ''
  d2$f2[is.na(d2$f2)] <- ''
  if(vn1=='dummy'){
    d2$f1 <- ''
  }
  
  tmp1 <- paste(d2$f1,d2$f2,sep='_')
  if(vn1=='dummy'){
    flag <- (d2$f2=='')
  }else{
    flag <- (d2$f1=='')&(d2$f2=='')
  }
  tmp1[flag] <- NA
  
  return(tmp1) 
} 


mlogloss <- function(pred_matrix, actual_m, cut_points) {
  
  pred_matrix <- pmin(pred_matrix, 1 - 10 ^ -15)
  pred_matrix <- pmax(pred_matrix, 10 ^ -15)
  pred_matrix <- pred_matrix / rowSums(pred_matrix)
  
  pred_matrix <- log(pred_matrix)
  
  ll <- sum(pred_matrix * actual_m)
  return(-1 * ll / nrow(actual_m))
}

char.to.int <- function(input){
  output=ifelse(input=='',NA,input)
  output=as.integer(factor(output,levels=sort(unique(output))))
  return(output)
}

sic.to.industry <- function(input){
  Industry <- rep(NA,length(input))
  Industry[input>99&input<=999&(!is.na(input))] <- 'Agriculture'
  Industry[input>999&input<=1499&(!is.na(input))] <- 'Mining'
  Industry[input>1499&input<=1799&(!is.na(input))] <- 'Construction'
  Industry[input>1999&input<=3999&(!is.na(input))] <- 'Manufacturing'
  Industry[input>3999&input<=4999&(!is.na(input))] <- 'Transportation'
  Industry[input>4999&input<=5199&(!is.na(input))] <- 'Wholesale'
  Industry[input>5199&input<=5999&(!is.na(input))] <- 'Retail'
  Industry[input>5999&input<=6799&(!is.na(input))] <- 'Finance'
  Industry[input>6999&input<=8999&(!is.na(input))] <- 'Service'
  Industry[input>9099&input<=9999&(!is.na(input))] <- 'Administration'
  return(Industry)
}
