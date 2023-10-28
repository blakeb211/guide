## Function for predicting least-squares mean of target
## Code produced by GUIDE 41.2 on 10/28/23 at 19:30
predicted <- function(){
 if(!is.na(num1) & num1 <= 1.50000000000 ){
   nodeid <- 2
   predict <- 3.50000000000
 } else {
   nodeid <- 3
   predict <- 11.0000000000
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
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
