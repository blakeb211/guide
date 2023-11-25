## Function for predicting least-squares mean of target
## Code produced by GUIDE 41.2 on 11/24/23 at 20:47
predicted <- function(){
 if(!is.na(num1) & num1 <= 3.85000000000 ){
   if(!is.na(num2) & num2 <= 5.50000000000 ){
     if(!is.na(num2) & num2 <= 3.50000000000 ){
       nodeid <- 8
       predict <- 6.32000000000
     } else {
       nodeid <- 9
       predict <- 8.60000000000
     }
   } else {
     if(!is.na(num1) & num1 <= 2.95000000000 ){
       nodeid <- 10
       predict <- 11.4800000000
     } else {
       nodeid <- 11
       predict <- 14.8000000000
     }
   }
 } else {
   if(!is.na(num2) & num2 <= 5.50000000000 ){
     if(!is.na(num2) & num2 <= 3.50000000000 ){
       nodeid <- 12
       predict <- 12.3200000000
     } else {
       nodeid <- 13
       predict <- 14.6000000000
     }
   } else {
     if(!is.na(num1) & num1 <= 5.95000000000 ){
       nodeid <- 14
       predict <- 17.3666666667
     } else {
       nodeid <- 15
       predict <- 21.3000000000
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
    num1 <- as.numeric(newdata$num1[i])
    num2 <- as.numeric(newdata$num2[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
