data.payment.summary <- function(train.prior.2015,slices){
  library(dplyr)
  library(data.table)
  library(moments)
  source('R/data.trim.zero.r')
  
  strt <- Sys.time()
  slice <- dplyr::slice
  
  train.prior.2015$n_mc <- vapply(strsplit(train.prior.2015$mc_string,'\\W+'),length,integer(1))
  
  # cut of years and put together
  train.prior.2015 <- merge(train.prior.2015,slices)
  train.prior.2015 <- mutate(train.prior.2015,
                                  slice=ifelse(slice>10,slice-10,slice),
                                  cutoff_year=2015-slice
  )
  train.prior.2015 <- train.prior.2015 %>% filter(paid_year<=cutoff_year)
  data <- data.trim.zero(train.prior.2015)
  
  # data <- NULL
  # for (cutoff in 1995:2014){
  #   tmp <- train.prior.2015.trim
  #   tmp$cutoff_year <- cutoff
  #   data <- rbind(data,tmp)
  # }
  # data <- data %>% filter(paid_year<=cutoff_year)
  
  
  # buckets <- c(-Inf, 0, 1000, 1400, 2000, 2800, 4000, 5750, 8000, 11500, 16250, 23250, 33000, 
  #              47000, 66000, 94000, 133000, 189000, 268000, 381000, 540000, Inf)
  # data$bin <- as.integer(cut(data$target, buckets)) - 1
  
  # onlevel payments
  # econ.2015 <- select(train.2015,econ_price_py,econ_price_healthcare_py) %>% distinct()
  # data <- merge(data,select(claims,claim_id,law_cola))
  # data <- mutate(data,
  #                paid_i=paid_i/econ_price_py*econ.2015$econ_price_py,
  #                # paid_i=ifelse(law_cola=='yes',paid_i/econ_price_py*econ.2015$econ_price_py,paid_i),
  #                paid_m=paid_m/econ_price_healthcare_py*econ.2015$econ_price_healthcare_py
  #                
  # )
  # data$target <- data$paid_i+data$paid_m

  # payments summary
  payment.summary <- data %>% 
    group_by(claim_id,cutoff_year) %>% 
    summarise(last_year=max(paid_year),
              first_year=min(paid_year),
              last_year_m=max(paid_year[paid_m>0]),
              last_year_i=max(paid_year[paid_i>0]),
              n_payment=max(paid_year)-min(paid_year)+1,
              avg_payment=mean(target),
              std_payment=sd(target),
              min_payment=min(target),
              max_payment=max(target),
              n_mean_mc=sum(n_mc),
              avg_payment_i=mean(paid_i),
              std_payment_i=sd(paid_i),
              min_payment_i=min(paid_i),
              max_payment_i=max(paid_i),
              avg_payment_m=mean(paid_m),
              std_payment_m=sd(paid_m),
              min_payment_m=min(paid_m),
              max_payment_m=max(paid_m),
              n_adj=n_distinct(adj),
              # avg_bin=mean(bin),
              # avg_bin_cnt_0=sum(bin==0)/n(),
              # avg_bin_cnt_1=sum(bin==1)/n(),
              # avg_bin_cnt_2=sum(bin==2)/n(),
              # avg_bin_cnt_3=sum(bin==3)/n(),
              # avg_bin_cnt_4=sum(bin==4)/n(),
              # avg_bin_cnt_5=sum(bin==5)/n(),
              # avg_bin_cnt_6=sum(bin==6)/n(),
              # avg_bin_cnt_7=sum(bin==7)/n(),
              # avg_bin_cnt_8=sum(bin==8)/n(),
              # avg_bin_cnt_9=sum(bin==9)/n(),
              # avg_bin_cnt_10_up=sum(bin>9)/n(),
              # avg_pay_gdp=mean(target/econ_gdp_py),
              # avg_pay_price=mean(target/econ_price_py),
              # avg_pay_price_allitem=mean(target/econ_price_allitems_py),
              # avg_pay_price_healthcare=mean(target/econ_price_healthcare_py),
              zero_precent=sum(target==0)/n()
    )
  
  last.payment <- data %>% arrange(claim_id,cutoff_year,desc(paid_year)) %>% 
    group_by(claim_id,cutoff_year) %>% slice(1) %>% 
    select(claim_id,cutoff_year,last_payment=target,last_mc=mc_string,last_adj=adj,
           n_last_mc=n_mc,
           last_payment_i=paid_i,
           last_payment_m=paid_m,
           last_i_last=last_i,
           last_m_last=last_m,
           last_l_last=last_l,
           last_o_last=last_o,
           
           last_gdp=econ_gdp_py,last_price=econ_price_py,
           last_price_allitems=econ_price_allitems_py,
           last_price_healthcare=econ_price_healthcare_py)
  last.payment$last_milo_last <- pmax(last.payment$last_m_last,last.payment$last_i_last,last.payment$last_l_last,last.payment$last_o_last)
  
  first.payment <- data %>% arrange(claim_id,cutoff_year,paid_year) %>% 
    group_by(claim_id,cutoff_year) %>% slice(1) %>% 
    select(claim_id,cutoff_year,first_payment=target,first_mc=mc_string,first_adj=adj,
           first_gdp=econ_gdp_py,first_price=econ_price_py,
           first_price_allitems=econ_price_allitems_py,
           first_price_healthcare=econ_price_healthcare_py)
  
  last.3.payment <- data %>% filter(paid_year>(cutoff_year-3)) %>% 
    group_by(claim_id,cutoff_year) %>% 
    summarise(last_3_payment_total=sum(target),n_last_3_mc=sum(n_mc),
              last_3_payment_i_total=sum(paid_i),
              last_3_payment_m_total=sum(paid_m)
              # last_3_bin_total=sum(bin)
              )
  
  last.5.payment <- data %>% filter(paid_year>(cutoff_year-5)) %>% 
    group_by(claim_id,cutoff_year) %>% 
    summarise(last_5_payment_total=sum(target),n_last_5_mc=sum(n_mc),
              last_5_payment_i_total=sum(paid_i),
              last_5_payment_m_total=sum(paid_m))
  
  payment.summary <- merge(payment.summary,last.payment,all=TRUE)
  payment.summary <- merge(payment.summary,first.payment,all=TRUE)
  payment.summary <- merge(payment.summary,last.3.payment,all=TRUE)
  payment.summary <- merge(payment.summary,last.5.payment,all=TRUE)
  
  payment.summary <- mutate(payment.summary,
                            total_payment=avg_payment*n_payment,
                            total_payment_i=avg_payment_i*n_payment,
                            total_payment_m=avg_payment_m*n_payment,
                            last_before_cutoff=cutoff_year-last_year,
                            # total_bin=avg_bin*n_payment,
                            n_total_mc=n_mean_mc*n_payment,
                            last_year=ifelse(last_year==cutoff_year,NA,last_year),
                            last_year_m=ifelse(last_year_m==cutoff_year,NA,last_year_m),
                            last_year_i=ifelse(last_year_i==cutoff_year,NA,last_year_i),
                            last_5_payment_total=ifelse(is.na(last_5_payment_total),0,last_5_payment_total),
                            last_5_payment_i_total=ifelse(is.na(last_5_payment_i_total),0,last_5_payment_i_total),
                            last_5_payment_m_total=ifelse(is.na(last_5_payment_m_total),0,last_5_payment_m_total),
                            last_5_payment_avg=last_5_payment_total/pmax(5-last_before_cutoff,0.01),
                            last_5_payment_i_avg=last_5_payment_i_total/pmax(5-last_before_cutoff,0.01),
                            last_5_payment_m_avg=last_5_payment_m_total/pmax(5-last_before_cutoff,0.01),
                            n_last_5_mc=ifelse(is.na(n_last_3_mc),0,n_last_5_mc),
                            # last_3_bin_avg=last_3_bin_total/max(3-last_before_cutoff,0.01),
                            last_3_payment_total=ifelse(is.na(last_3_payment_total),0,last_3_payment_total),
                            last_3_payment_i_total=ifelse(is.na(last_3_payment_i_total),0,last_3_payment_i_total),
                            last_3_payment_m_total=ifelse(is.na(last_3_payment_m_total),0,last_3_payment_m_total),
                            n_last_3_mc=ifelse(is.na(n_last_3_mc),0,n_last_3_mc),
                            last_3_payment_avg=last_3_payment_total/pmax(3-last_before_cutoff,0.01),
                            last_3_payment_i_avg=last_3_payment_i_total/pmax(3-last_before_cutoff,0.01),
                            last_3_payment_m_avg=last_3_payment_m_total/pmax(3-last_before_cutoff,0.01)
  )
  
  print(Sys.time()-strt)
  return(payment.summary)
}
