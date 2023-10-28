## Function for predicting least-squares mean of target
## Code produced by GUIDE 41.2 on 10/28/23 at 19:36
predicted <- function(){
 catvalues <- c("3")
 if(cat1 %in% catvalues){
   if(!is.na(num1) & num1 <= 1.50000000000 ){
     nodeid <- 4
     predict <- 9.41568651168
   } else {
     nodeid <- 5
     predict <- 8.17231453924
   }
 } else {
   catvalues <- c("1")
   if(cat1 %in% catvalues){
     nodeid <- 6
     predict <- 2.55285864238
   } else {
     nodeid <- 7
     predict <- 4.62068508440
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
    cat1 <- as.character(newdata$cat1[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
