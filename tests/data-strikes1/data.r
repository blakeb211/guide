## Function for predicting least-squares mean of strikes
## Code produced by GUIDE 41.2 on 11/25/23 at 21:10
predicted <- function(){
 catvalues <- c("0.0","1.0")
 if(unemp %in% catvalues){
   if(!is.na(union) & union <= 0.812500000000 ){
     if(!is.na(union) & union <= 0.625000000000 ){
       nodeid <- 8
       predict <- 167.215053763
     } else {
       nodeid <- 9
       predict <- 261.157894737
     }
   } else {
     if(!is.na(union) & union <= 0.937500000000 ){
       nodeid <- 10
       predict <- 82.8666666667
     } else {
       nodeid <- 11
       predict <- 27.0000000000
     }
   }
 } else {
   if(!is.na(union) & union <= 0.625000000000 ){
     catvalues <- c("0.0")
     if(inflat %in% catvalues){
       nodeid <- 12
       predict <- 443.960526316
     } else {
       nodeid <- 13
       predict <- 591.594936709
     }
   } else {
     if(!is.na(politis) & politis <= 28.9000000000 ){
       nodeid <- 14
       predict <- 467.500000000
     } else {
       nodeid <- 15
       predict <- 98.1607142857
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
