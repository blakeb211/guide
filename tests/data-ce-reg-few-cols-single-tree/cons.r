## Function for predicting least-squares mean of INTRDVX
## Code produced by GUIDE 41.2 on 9/9/23 at 17:4
predicted <- function(){
 if(!is.na(REFGEN) & REFGEN <= 4.50000000000 ){
   if(!is.na(INC_RANK) & INC_RANK <= 0.840186250000 ){
     catvalues <- c("1","4","6")
     if(INCNONW1 %in% catvalues){
       nodeid <- 8
       predict <- 4147.85464809
     } else {
       nodeid <- 9
       predict <- 1416.34186107
     }
   } else {
     catvalues <- c("1")
     if(INCNONW1 %in% catvalues){
       nodeid <- 10
       predict <- 41147.2485814
     } else {
       nodeid <- 11
       predict <- 9023.60503037
     }
   }
 } else {
   if(!is.na(INC_RANK) & INC_RANK <= 0.936266900000 ){
     if(!is.na(INC_RANK) & INC_RANK <= 0.818551050000 ){
       nodeid <- 12
       predict <- 394.089385435
     } else {
       nodeid <- 13
       predict <- 899.969487084
     }
   } else {
     if(!is.na(AGE_REF) & AGE_REF <= 34.5000000000 ){
       nodeid <- 14
       predict <- 1091.91271527
     } else {
       nodeid <- 15
       predict <- 6061.89855279
     }
   }
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
    AGE_REF <- as.numeric(newdata$AGE_REF[i])
    INC_RANK <- as.numeric(newdata$INC_RANK[i])
    INCNONW1 <- as.character(newdata$INCNONW1[i])
    REFGEN <- as.numeric(newdata$REFGEN[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
