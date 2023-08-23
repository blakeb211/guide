## Function for predicting least-squares mean of INTRDVX
## Code produced by GUIDE 41.2 on 8/20/23 at 15:51
predicted <- function(){
 if(!is.na(REFGEN) & REFGEN <= 4.50000000000 ){
   if(!is.na(INC_RANK) & INC_RANK <= 0.840186250000 ){
     nodeid <- 4
     predict <- 2822.64453247
   } else {
     catvalues <- c("8")
     if(EARNCOMP %in% catvalues){
       catvalues <- c("1")
       if(RETSURV %in% catvalues){
         nodeid <- 20
         predict <- 27641.2815914
       } else {
         nodeid <- 21
         predict <- 85859.2758334
       }
     } else {
       if(!is.na(FFTAXOWE) & FFTAXOWE <= 27769.5000000 ){
         nodeid <- 22
         predict <- 2646.53671681
       } else {
         if(is.na(AGE2) | AGE2 <= 56.5000000000 ){
           nodeid <- 46
           predict <- 8036.33405691
         } else {
           if(!is.na(BATHRMQ) & BATHRMQ <= 2.50000000000 ){
             nodeid <- 94
             predict <- 10866.5204904
           } else {
             nodeid <- 95
             predict <- 46702.3978612
           }
         }
       }
     }
   }
 } else {
   nodeid <- 3
   predict <- 980.352916577
 }
 return(c(nodeid,predict))
}
## end of function
##
##
## If desired, replace "ce2021.txt" with name of file containing new data
## New file must have at least the same variables with same names
## (but not necessarily the same order) as in the training data file
## Missing value code is converted to NA if not already NA
newdata <- read.table("ce2021.txt",header=TRUE,colClasses="character")
## node contains terminal node ID of each case
## pred contains predicted value of each case
node <- NULL
pred <- NULL
for(i in 1:nrow(newdata)){
    AGE2 <- as.numeric(newdata$AGE2[i])
    BATHRMQ <- as.numeric(newdata$BATHRMQ[i])
    EARNCOMP <- as.character(newdata$EARNCOMP[i])
    INC_RANK <- as.numeric(newdata$INC_RANK[i])
    RETSURV <- as.character(newdata$RETSURV[i])
    REFGEN <- as.numeric(newdata$REFGEN[i])
    FFTAXOWE <- as.numeric(newdata$FFTAXOWE[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
