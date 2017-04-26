data.payment.history <- function(train.prior.2015,slices){
  library(dplyr)
  library(data.table)
  library(moments)
  
  strt <- Sys.time()
  slice <- dplyr::slice
  
  # cut of years and put together
  train.prior.2015 <- merge(train.prior.2015,slices)
  train.prior.2015 <- mutate(train.prior.2015,
                                  slice=ifelse(slice>10,slice-10,slice),
                                  cutoff_year=2015-slice
  )
  data <- train.prior.2015 %>% filter(paid_year<=cutoff_year)
  
  
  # data <- NULL
  # for (cutoff in 1995:2014){
  #   tmp <- train.prior.2015
  #   tmp$cutoff_year <- cutoff
  #   data <- rbind(data,tmp)
  # }
  # data <- data %>% filter(paid_year<=cutoff_year)
  
  
  data$p <- 'p'
  data$i <- 'i'
  data$m <- 'm'
  # data$p_paid_year <- paste0(data$p,data$paid_year)
  # data$p_paid_year_i <- paste0(data$p_paid_year,data$i)
  # data$p_paid_year_m <- paste0(data$p_paid_year,data$m)
  # payment.history <- select(data,claim_id,cutoff_year,p_paid_year,target) %>% dcast(formula=claim_id+cutoff_year~p_paid_year,value.var='target')
  # payment.history_i <- select(data,claim_id,cutoff_year,p_paid_year_i,paid_i) %>% dcast(formula=claim_id+cutoff_year~p_paid_year_i,value.var='paid_i')
  # payment.history_m <- select(data,claim_id,cutoff_year,p_paid_year_m,paid_m) %>% dcast(formula=claim_id+cutoff_year~p_paid_year_m,value.var='paid_m')
  
  data <- filter(data,cutoff_year-paid_year<10)
  data$p_paid_bef_cut <- paste0(data$p,data$cutoff_year-data$paid_year)
  data$p_paid_bef_cut_i <- paste0(data$p_paid_bef_cut,data$i)
  data$p_paid_bef_cut_m <- paste0(data$p_paid_bef_cut,data$m)
  payment.history <- select(data,claim_id,cutoff_year,p_paid_bef_cut,target) %>% dcast(formula=claim_id+cutoff_year~p_paid_bef_cut,value.var='target')
  payment.history.last_i <- select(data,claim_id,cutoff_year,p_paid_bef_cut_i,paid_i) %>% dcast(formula=claim_id+cutoff_year~p_paid_bef_cut_i,value.var='paid_i')
  payment.history.last_m <- select(data,claim_id,cutoff_year,p_paid_bef_cut_m,paid_m) %>% dcast(formula=claim_id+cutoff_year~p_paid_bef_cut_m,value.var='paid_m')
  
  # payment.history <- merge(payment.history,payment.history_i,all=TRUE)
  # payment.history <- merge(payment.history,payment.history_m,all=TRUE)
  # payment.history <- merge(payment.history,payment.history.last,all=TRUE)
  payment.history <- merge(payment.history,payment.history.last_i,all=TRUE)
  payment.history <- merge(payment.history,payment.history.last_m,all=TRUE)
  
  return(payment.history)
}