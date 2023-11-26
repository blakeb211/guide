## Function for predicting least-squares mean of tgt
## Code produced by GUIDE 41.2 on 11/26/23 at 15:46
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
           catvalues <- c("H")
           if(cat1 %in% catvalues){
             nodeid <- 68
             predict <- 9767.12727273
           } else {
             catvalues <- c("A")
             if(cat1 %in% catvalues){
               nodeid <- 138
               predict <- 9379.50000000
             } else {
               nodeid <- 139
               predict <- 9463.81421053
             }
           }
         } else {
           catvalues <- c("C","F")
           if(cat1 %in% catvalues){
             catvalues <- c("C")
             if(cat1 %in% catvalues){
               nodeid <- 140
               predict <- 4575.78733333
             } else {
               nodeid <- 141
               predict <- 5514.13333333
             }
           } else {
             catvalues <- c("D")
             if(cat1 %in% catvalues){
               nodeid <- 142
               predict <- 7739.42000000
             } else {
               catvalues <- c("I")
               if(cat1 %in% catvalues){
                 nodeid <- 286
                 predict <- 7166.10263158
               } else {
                 catvalues <- c("B")
                 if(cat1 %in% catvalues){
                   nodeid <- 574
                   predict <- 6834.08000000
                 } else {
                   nodeid <- 575
                   predict <- 6706.66200000
                 }
               }
             }
           }
         }
       }
     } else {
       if(!is.na(num2) & num2 <= 0.625000000000E-1 ){
         catvalues <- c("G","I","J")
         if(cat1 %in% catvalues){
           nodeid <- 36
           predict <- 27137.5809091
         } else {
           catvalues <- c("A","D","F")
           if(cat1 %in% catvalues){
             nodeid <- 74
             predict <- 7880.23666667
           } else {
             nodeid <- 75
             predict <- 16105.0209091
           }
         }
       } else {
         catvalues <- c("E","H","I")
         if(cat1 %in% catvalues){
           if(!is.na(num2) & num2 <= 0.312500000000 ){
             nodeid <- 76
             predict <- 14437.3111111
           } else {
             nodeid <- 77
             predict <- 16318.5783333
           }
         } else {
           if(!is.na(num2) & num2 <= 0.187500000000 ){
             nodeid <- 78
             predict <- 6106.91727273
           } else {
             if(!is.na(num2) & num2 <= 0.312500000000 ){
               catvalues <- c("B","G")
               if(cat1 %in% catvalues){
                 nodeid <- 316
                 predict <- 9816.39142857
               } else {
                 nodeid <- 317
                 predict <- 4554.61125000
               }
             } else {
               catvalues <- c("A","B","D")
               if(cat1 %in% catvalues){
                 nodeid <- 318
                 predict <- 6639.75333333
               } else {
                 nodeid <- 319
                 predict <- 8936.87333333
               }
             }
           }
         }
       }
     }
   } else {
     if(!is.na(num2) & num2 <= 0.812500000000 ){
       catvalues <- c("A","E","F","G")
       if(cat1 %in% catvalues){
         catvalues <- c("A","E")
         if(cat1 %in% catvalues){
           catvalues <- c("A")
           if(cat1 %in% catvalues){
             nodeid <- 80
             predict <- 3638.19444444
           } else {
             nodeid <- 81
             predict <- 4189.68000000
           }
         } else {
           nodeid <- 41
           predict <- 2619.75444444
         }
       } else {
         catvalues <- c("B","C","J")
         if(cat1 %in% catvalues){
           catvalues <- c("B")
           if(cat1 %in% catvalues){
             nodeid <- 84
             predict <- 8434.30500000
           } else {
             catvalues <- c("C")
             if(cat1 %in% catvalues){
               nodeid <- 170
               predict <- 9707.46250000
             } else {
               nodeid <- 171
               predict <- 9644.60500000
             }
           }
         } else {
           if(!is.na(num2) & num2 <= 0.625000000000 ){
             nodeid <- 86
             predict <- 6594.42000000
           } else {
             nodeid <- 87
             predict <- 6948.06500000
           }
         }
       }
     } else {
       catvalues <- c("D","J")
       if(cat1 %in% catvalues){
         nodeid <- 22
         predict <- 8005.46285714
       } else {
         if(!is.na(num2) & num2 <= 0.937500000000 ){
           catvalues <- c("A","C","E","G")
           if(cat1 %in% catvalues){
             nodeid <- 92
             predict <- 233.261428571
           } else {
             nodeid <- 93
             predict <- 1667.51416667
           }
         } else {
           nodeid <- 47
           predict <- 1453.96692308
         }
       }
     }
   }
 } else {
   if(!is.na(num2) & num2 <= 1.62500000000 ){
     if(!is.na(num2) & num2 <= 0.875000000000 ){
       if(!is.na(num2) & num2 <= 0.375000000000 ){
         if(!is.na(num2) & num2 <= 0.125000000000 ){
           catvalues <- c("A","E","H","I")
           if(cat1 %in% catvalues){
             nodeid <- 96
             predict <- 9995.95800000
           } else {
             catvalues <- c("B","D","J")
             if(cat1 %in% catvalues){
               nodeid <- 194
               predict <- 16687.9040000
             } else {
               nodeid <- 195
               predict <- 21771.9750000
             }
           }
         } else {
           nodeid <- 49
           predict <- 7656.77400000
         }
       } else {
         if(!is.na(num2) & num2 <= 0.625000000000 ){
           catvalues <- c("C","G","I")
           if(cat1 %in% catvalues){
             nodeid <- 100
             predict <- 29824.5950000
           } else {
             nodeid <- 101
             predict <- 17025.6277778
           }
         } else {
           catvalues <- c("D","G","H")
           if(cat1 %in% catvalues){
             nodeid <- 102
             predict <- 22477.7614286
           } else {
             catvalues <- c("B","E")
             if(cat1 %in% catvalues){
               nodeid <- 206
               predict <- 14082.5250000
             } else {
               catvalues <- c("J")
               if(cat1 %in% catvalues){
                 nodeid <- 414
                 predict <- 16498.4057143
               } else {
                 nodeid <- 415
                 predict <- 18369.8246154
               }
             }
           }
         }
       }
     } else {
       if(!is.na(num2) & num2 <= 1.25000000000 ){
         catvalues <- c("A","C","D","J")
         if(cat1 %in% catvalues){
           catvalues <- c("A","D")
           if(cat1 %in% catvalues){
             nodeid <- 104
             predict <- 8732.91571429
           } else {
             nodeid <- 105
             predict <- 3360.06444444
           }
         } else {
           catvalues <- c("G","H")
           if(cat1 %in% catvalues){
             nodeid <- 106
             predict <- 23166.8122222
           } else {
             nodeid <- 107
             predict <- 16249.6309091
           }
         }
       } else {
         catvalues <- c("A","E")
         if(cat1 %in% catvalues){
           nodeid <- 54
           predict <- 18294.5562500
         } else {
           catvalues <- c("B","D","F","H")
           if(cat1 %in% catvalues){
             catvalues <- c("B","F")
             if(cat1 %in% catvalues){
               nodeid <- 220
               predict <- 9904.00000000
             } else {
               nodeid <- 221
               predict <- 10867.0000000
             }
           } else {
             nodeid <- 111
             predict <- 6041.46400000
           }
         }
       }
     }
   } else {
     if(!is.na(num2) & num2 <= 1.87500000000 ){
       catvalues <- c("B","D","I")
       if(cat1 %in% catvalues){
         nodeid <- 28
         predict <- 1289.11111111
       } else {
         catvalues <- c("C","G","J")
         if(cat1 %in% catvalues){
           nodeid <- 58
           predict <- 3050.07100000
         } else {
           nodeid <- 59
           predict <- 4658.15818182
         }
       }
     } else {
       nodeid <- 15
       predict <- 1034.69285714
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
    cat1 <- as.character(newdata$cat1[i])
    num1 <- as.numeric(newdata$num1[i])
    num2 <- as.numeric(newdata$num2[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
