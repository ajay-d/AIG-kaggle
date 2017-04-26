data.last.mc.mat <- function(payment.summary){
mc <- strsplit(payment.summary$last_mc,'\\W+')
uniq <- sort(unique(unlist(mc)))
dmat <- matrix(0,ncol=length(uniq),nrow=nrow(payment.summary))
colnames(dmat) <- uniq

for (i in 1:nrow(payment.summary)){
  if(length(mc[[i]])>0){
    if (!is.na(mc[[i]])){
      dmat[i,mc[[i]]] <- 1
    }
  }
}

last.mc.mat <- cbind(select(payment.summary,claim_id,cutoff_year),dmat)
return(last.mc.mat)
}

