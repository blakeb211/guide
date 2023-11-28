## Function for predicting least-squares mean of tgt
## Code produced by GUIDE 41.2 on 11/27/23 at 20:2
predicted <- function(){
 if(!is.na(num1) & num1 <= 1.50000000000 ){
   if(!is.na(num2) & num2 <= 0.437500000000 ){
     if(!is.na(num1) & num1 <= 0.500000000000 ){
       catvalues <- c("J")
       if(cat1 %in% catvalues){
         nodeid <- 16
         predict <- 14041.2193103
       } else {
         catvalues <- c("A","G","H")
         if(cat1 %in% catvalues){
           catvalues <- c("A")
           if(cat1 %in% catvalues){
             nodeid <- 68
             predict <- 9379.50000000
           } else {
             nodeid <- 69
             predict <- 9626.56756098
           }
         } else {
           catvalues <- c("C","F")
           if(cat1 %in% catvalues){
             nodeid <- 70
             predict <- 5044.96033333
           } else {
             catvalues <- c("B","E")
             if(cat1 %in% catvalues){
               nodeid <- 142
               predict <- 6788.57357143
             } else {
               nodeid <- 143
               predict <- 7486.09372093
             }
           }
         }
       }
     } else {
       if(!is.na(num2) & num2 <= 0.625000000000E-1 ){
         nodeid <- 18
         predict <- 17631.9596774
       } else {
         catvalues <- c("A","B","D","G","J")
         if(cat1 %in% catvalues){
           nodeid <- 38
           predict <- 6864.14658537
         } else {
           nodeid <- 39
           predict <- 13954.2722222
         }
       }
     }
   } else {
     if(!is.na(num2) & num2 <= 0.812500000000 ){
       catvalues <- c("A","D","E","F","G")
       if(cat1 %in% catvalues){
         nodeid <- 20
         predict <- 3731.17407407
       } else {
         nodeid <- 21
         predict <- 8295.98325000
       }
     } else {
       nodeid <- 11
       predict <- 2476.48230769
     }
   }
 } else {
   if(!is.na(num2) & num2 <= 1.62500000000 ){
     if(!is.na(num2) & num2 <= 0.875000000000 ){
       if(!is.na(num2) & num2 <= 0.375000000000 ){
         nodeid <- 24
         predict <- 13977.2483721
       } else {
         catvalues <- c("A","B","C","E","J")
         if(cat1 %in% catvalues){
           nodeid <- 50
           predict <- 16289.9088462
         } else {
           nodeid <- 51
           predict <- 22725.3090909
         }
       }
     } else {
       if(!is.na(num2) & num2 <= 1.25000000000 ){
         nodeid <- 26
         predict <- 13294.9511111
       } else {
         nodeid <- 27
         predict <- 11001.0633333
       }
     }
   } else {
     nodeid <- 7
     predict <- 2718.52162162
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
    cat1 <- as.character(newdata$cat1[i])
    num1 <- as.numeric(newdata$num1[i])
    num2 <- as.numeric(newdata$num2[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
