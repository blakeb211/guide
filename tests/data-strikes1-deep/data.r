## Function for predicting least-squares mean of strikes
## Code produced by GUIDE 41.2 on 11/28/23 at 7:18
predicted <- function(){
 catvalues <- c("0.0","1.0")
 if(unemp %in% catvalues){
   if(!is.na(union) & union <= 0.812500000000 ){
     if(!is.na(union) & union <= 0.625000000000 ){
       if(!is.na(union) & union <= 0.437500000000 ){
         nodeid <- 16
         predict <- 206.129496403
       } else {
         nodeid <- 17
         predict <- 52.1276595745
       }
     } else {
       if(!is.na(politis) & politis <= 28.6000000000 ){
         nodeid <- 18
         predict <- 502.000000000
       } else {
         nodeid <- 19
         predict <- 120.666666667
       }
     }
   } else {
     if(!is.na(union) & union <= 0.937500000000 ){
       if(!is.na(politis) & politis <= 52.1000000000 ){
         nodeid <- 20
         predict <- 61.1111111111
       } else {
         nodeid <- 21
         predict <- 148.133333333
       }
     } else {
       if(!is.na(politis) & politis <= 47.6000000000 ){
         nodeid <- 22
         predict <- 75.0000000000
       } else {
         nodeid <- 23
         predict <- 9.00000000000
       }
     }
   }
 } else {
   if(!is.na(union) & union <= 0.625000000000 ){
     catvalues <- c("0.0")
     if(inflat %in% catvalues){
       if(!is.na(union) & union <= 0.625000000000E-1 ){
         nodeid <- 24
         predict <- 596.781250000
       } else {
         nodeid <- 25
         predict <- 332.818181818
       }
     } else {
       catvalues <- c("2.0")
       if(unemp %in% catvalues){
         nodeid <- 26
         predict <- 717.068965517
       } else {
         nodeid <- 27
         predict <- 518.820000000
       }
     }
   } else {
     if(!is.na(politis) & politis <= 28.9000000000 ){
       nodeid <- 14
       predict <- 467.500000000
     } else {
       if(!is.na(union) & union <= 0.812500000000 ){
         nodeid <- 30
         predict <- 138.484848485
       } else {
         nodeid <- 31
         predict <- 40.3043478261
       }
     }
   }
 }
 return(c(nodeid,predict))
}
## end of function
##
##
## If desired, replace "data.txt" with name of file containing new data
## New file must have at least the same variables with same names
## (but not necessarily the same order) as in the training data file
## Missing value code is converted to NA if not already NA
newdata <- read.table("data.txt",header=TRUE,colClasses="character")
## node contains terminal node ID of each case
## pred contains predicted value of each case
node <- NULL
pred <- NULL
for(i in 1:nrow(newdata)){
    unemp <- as.character(newdata$unemp[i])
    inflat <- as.character(newdata$inflat[i])
    politis <- as.numeric(newdata$politis[i])
    union <- as.numeric(newdata$union[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
