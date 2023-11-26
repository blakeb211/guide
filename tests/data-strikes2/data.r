## Function for predicting least-squares mean of strikes
## Code produced by GUIDE 41.2 on 11/26/23 at 10:53
predicted <- function(){
 catvalues <- c("0.0","1.0")
 if(unemp %in% catvalues){
   if(!is.na(union) & union <= 0.812500000000 ){
     nodeid <- 4
     predict <- 189.251028807
   } else {
     nodeid <- 5
     predict <- 67.8780487805
   }
 } else {
   if(!is.na(union) & union <= 0.625000000000 ){
     nodeid <- 6
     predict <- 543.645299145
   } else {
     nodeid <- 7
     predict <- 154.121212121
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
    union <- as.numeric(newdata$union[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
