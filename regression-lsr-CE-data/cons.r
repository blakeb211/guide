## Function for predicting least-squares mean of INTRDVX
## Code produced by GUIDE 41.2 on 9/6/23 at 16:4
predicted <- function(){
 if(!is.na(REFGEN) & REFGEN <= 4.50000000000 ){
   if(!is.na(INC_RANK) & INC_RANK <= 0.840186250000 ){
     catvalues <- c("2","4")
     if(REF_RACE %in% catvalues){
       catvalues <- c("S12A","S12B","S23A","S24B","S35C","S37B","S48A","S49D")
       if(PSU %in% catvalues){
         nodeid <- 16
         predict <- 1018.30576700
       } else {
         if(!is.na(HLFBATHQ) & HLFBATHQ <= 0.500000000000 ){
           if(!is.na(ESHELTRC) & ESHELTRC <= 1047.33350000 ){
             nodeid <- 68
             predict <- 163.995273833
           } else {
             nodeid <- 69
             predict <- 26.6585853500
           }
         } else {
           nodeid <- 35
           predict <- 1097.20822589
         }
       }
     } else {
       catvalues <- c("1")
       if(REF_RACE %in% catvalues){
         if(is.na(STOCKX) | STOCKX <= 59950.0000000 ){
           if(!is.na(MRTINTPQ) & MRTINTPQ <= 1.83335000000 ){
             if(!is.na(FINCBTAX) & FINCBTAX <= 106565.500000 ){
               if(!is.na(FINCBTAX) & FINCBTAX <= 49134.5000000 ){
                 catvalues <- c("13","18","19","27","31","33","40","41","47","54")
                 if(STATE %in% catvalues){
                   if(!is.na(HIGH_EDU) & HIGH_EDU <= 13.5000000000 ){
                     nodeid <- 1152
                     predict <- 60.6553131103
                   } else {
                     nodeid <- 1153
                     predict <- 250.204224866
                   }
                 } else {
                   catvalues <- c("4","6","9","20","36","48","49","55")
                   if(STATE %in% catvalues){
                     if(!is.na(HIGH_EDU) & HIGH_EDU <= 13.5000000000 ){
                       nodeid <- 2308
                       predict <- 876.201377613
                     } else {
                       catvalues <- c("1")
                       if(RACE2 %in% catvalues){
                         nodeid <- 4618
                         predict <- 588.431179516
                       } else {
                         nodeid <- 4619
                         predict <- 5468.63875109
                       }
                     }
                   } else {
                     if(!is.na(EDUCA2) & EDUCA2 <= 13.5000000000 ){
                       nodeid <- 2310
                       predict <- 358.515580054
                     } else {
                       if(!is.na(FRRETIRM) & FRRETIRM <= 24439.0000000 ){
                         nodeid <- 4622
                         predict <- 1164.09134235
                       } else {
                         nodeid <- 4623
                         predict <- 3011.34113123
                       }
                     }
                   }
                 }
               } else {
                 if(is.na(AGE2) | AGE2 <= 76.5000000000 ){
                   if(!is.na(VEHQ) & VEHQ <= 2.50000000000 ){
                     if(!is.na(FSALARYX) & FSALARYX <= 1750.00000000 ){
                       if(!is.na(FFTAXOWE) & FFTAXOWE <= 2058.00000000 ){
                         nodeid <- 4624
                         predict <- 3053.00579158
                       } else {
                         nodeid <- 4625
                         predict <- 5648.22473267
                       }
                     } else {
                       if(!is.na(WATRPSPQ) & WATRPSPQ <= 49.5000000000 ){
                         nodeid <- 4626
                         predict <- 116.412871505
                       } else {
                         nodeid <- 4627
                         predict <- 1974.83169477
                       }
                     }
                   } else {
                     if(!is.na(INC_RANK) & INC_RANK <= 0.580003600000 ){
                       nodeid <- 2314
                       predict <- 1791.72107966
                     } else {
                       if(!is.na(SHELTCQ) & SHELTCQ <= 407.500000000 ){
                         nodeid <- 4630
                         predict <- 4424.90288614
                       } else {
                         nodeid <- 4631
                         predict <- 8065.56332134
                       }
                     }
                   }
                 } else {
                   if(!is.na(ETOTALC) & ETOTALC <= 4565.66650000 ){
                     nodeid <- 1158
                     predict <- 13515.5117036
                   } else {
                     nodeid <- 1159
                     predict <- 7175.30970834
                   }
                 }
               }
             } else {
               if(!is.na(PERINSPQ) & PERINSPQ <= 848.191700000 ){
                 nodeid <- 290
                 predict <- 24297.9851763
               } else {
                 nodeid <- 291
                 predict <- 5740.33047075
               }
             }
           } else {
             if(!is.na(RETSURVX) & RETSURVX <= 3650.00000000 ){
               if(!is.na(BUILT) & BUILT <= 1988.00000000 ){
                 nodeid <- 292
                 predict <- 2048.66123802
               } else {
                 nodeid <- 293
                 predict <- 3207.54771266
               }
             } else {
               if(!is.na(PERSOT64) & PERSOT64 <= 0.500000000000 ){
                 catvalues <- c("3","8","9")
                 if(UNISTRQ %in% catvalues){
                   catvalues <- c("9","12","17","24","25","26","32","34","41","53")
                   if(STATE %in% catvalues){
                     nodeid <- 1176
                     predict <- 58.8374428104
                   } else {
                     catvalues <- c("3","4","5","6")
                     if(DIVISION %in% catvalues){
                       if(!is.na(PROPTXPQ) & PROPTXPQ <= 487.500000000 ){
                         nodeid <- 4708
                         predict <- 92.0332554576
                       } else {
                         nodeid <- 4709
                         predict <- 965.426020050
                       }
                     } else {
                       if(!is.na(ALCBEVPQ) & ALCBEVPQ <= 22.5000000000 ){
                         nodeid <- 4710
                         predict <- 49.1516420450
                       } else {
                         nodeid <- 4711
                         predict <- 326.007456667
                       }
                     }
                   }
                 } else {
                   nodeid <- 589
                   predict <- 1416.47530007
                 }
               } else {
                 catvalues <- c("6","8","28","53","NA")
                 catvalues <- c(catvalues,NA)
                 if(is.na(STATE) | STATE %in% catvalues){
                   if(!is.na(NUM_AUTO) & NUM_AUTO <= 0.500000000000 ){
                     nodeid <- 1180
                     predict <- 300.949253314
                   } else {
                     nodeid <- 1181
                     predict <- 2996.66342957
                   }
                 } else {
                   if(!is.na(FRRETIRM) & FRRETIRM <= 1392.00000000 ){
                     nodeid <- 1182
                     predict <- 7282.37157745
                   } else {
                     if(!is.na(NUM_AUTO) & NUM_AUTO <= 0.500000000000 ){
                       if(!is.na(ELCTRCPQ) & ELCTRCPQ <= 201.500000000 ){
                         nodeid <- 4732
                         predict <- 398.516757277
                       } else {
                         nodeid <- 4733
                         predict <- 374.217868738
                       }
                     } else {
                       if(!is.na(FOODCQ) & FOODCQ <= 630.500000000 ){
                         nodeid <- 4734
                         predict <- 485.609932407
                       } else {
                         nodeid <- 4735
                         predict <- 1452.12285975
                       }
                     }
                   }
                 }
               }
             }
           }
         } else {
           nodeid <- 37
           predict <- 9803.02222672
         }
       } else {
         catvalues <- c("2","6")
         if(STATE %in% catvalues){
           nodeid <- 38
           predict <- 3377.37843799
         } else {
           nodeid <- 39
           predict <- 8154.15851958
         }
       }
     }
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
         catvalues <- c("4","5","6")
         if(UNISTRQ %in% catvalues){
           nodeid <- 44
           predict <- 6613.21335455
         } else {
           catvalues <- c("S49A","S49B","S49D","S49E","S49G")
           if(PSU %in% catvalues){
             nodeid <- 90
             predict <- 1207.38507426
           } else {
             if(!is.na(FJSSDEDX) & FJSSDEDX <= 15951.0000000 ){
               catvalues <- c("1","4","9","12","25","31","40","42","48","49","51","53","NA")
               catvalues <- c(catvalues,NA)
               if(is.na(STATE) | STATE %in% catvalues){
                 catvalues <- c("3","NA")
                 catvalues <- c(catvalues,NA)
                 if(is.na(REGION) | REGION %in% catvalues){
                   nodeid <- 728
                   predict <- 5495.01528112
                 } else {
                   catvalues <- c("3","6","NA")
                   catvalues <- c(catvalues,NA)
                   if(is.na(OCCUCOD2) | OCCUCOD2 %in% catvalues){
                     nodeid <- 1458
                     predict <- 429.363253411
                   } else {
                     nodeid <- 1459
                     predict <- 120.285924217
                   }
                 }
               } else {
                 catvalues <- c("2","3","5","8")
                 if(DIVISION %in% catvalues){
                   catvalues <- c("1","3","12")
                   if(OCCUCOD2 %in% catvalues){
                     nodeid <- 1460
                     predict <- 1600.37980556
                   } else {
                     nodeid <- 1461
                     predict <- 60.1850599834
                   }
                 } else {
                   nodeid <- 731
                   predict <- 1923.03606265
                 }
               }
             } else {
               nodeid <- 183
               predict <- 6627.86421504
             }
           }
         }
       } else {
         if(is.na(AGE2) | AGE2 <= 56.5000000000 ){
           if(!is.na(FINCBTAX) & FINCBTAX <= 444350.000000 ){
             if(!is.na(POPSIZE) & POPSIZE <= 1.50000000000 ){
               nodeid <- 184
               predict <- 9871.57592338
             } else {
               catvalues <- c("4","NA")
               catvalues <- c(catvalues,NA)
               if(is.na(RACE2) | RACE2 %in% catvalues){
                 nodeid <- 370
                 predict <- 7288.56893212
               } else {
                 if(!is.na(EOWNDWLC) & EOWNDWLC <= 1190.66700000 ){
                   nodeid <- 742
                   predict <- 4619.91704938
                 } else {
                   nodeid <- 743
                   predict <- 1766.84598670
                 }
               }
             }
           } else {
             nodeid <- 93
             predict <- 19967.9028959
           }
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
   catvalues <- c("S11A","S12A","S12B","S35C","S35D","S35E","S37B","S48A","S49B","S49F","S49G")
   if(PSU %in% catvalues){
     if(!is.na(INC_RANK) & INC_RANK <= 0.792165500000 ){
       if(!is.na(POPSIZE) & POPSIZE <= 1.50000000000 ){
         nodeid <- 24
         predict <- 481.594404127
       } else {
         catvalues <- c("S11A","S35D","S35E")
         if(PSU %in% catvalues){
           nodeid <- 50
           predict <- 468.009694112
         } else {
           nodeid <- 51
           predict <- 246.350478775
         }
       }
     } else {
       if(!is.na(LIFINSPQ) & LIFINSPQ <= 86.0000000000 ){
         if(!is.na(RETPENCQ) & RETPENCQ <= 1287.42500000 ){
           nodeid <- 52
           predict <- 1279.56906002
         } else {
           nodeid <- 53
           predict <- 2818.85777894
         }
       } else {
         nodeid <- 27
         predict <- 2638.28874803
       }
     }
   } else {
     catvalues <- c("4","6")
     if(CUTENURE %in% catvalues){
       if(is.na(LIQUIDX)){
         catvalues <- c("6","7")
         if(UNISTRQ %in% catvalues){
           nodeid <- 56
           predict <- 1577.27987289
         } else {
           if(!is.na(INCLASS2) & INCLASS2 <= 4.50000000000 ){
             catvalues <- c("3","6","7","9","10")
             if(OCCUCOD1 %in% catvalues){
               nodeid <- 228
               predict <- 26.3840652724
             } else {
               nodeid <- 229
               predict <- 97.4856172884
             }
           } else {
             catvalues <- c("4","10")
             if(UNISTRQ %in% catvalues){
               nodeid <- 230
               predict <- 78.5829120379
             } else {
               nodeid <- 231
               predict <- 611.351507622
             }
           }
         }
       } else {
         nodeid <- 29
         predict <- 266.798598920
       }
     } else {
       catvalues <- c("1")
       if(CUTENURE %in% catvalues){
         catvalues <- c("1","7","NA")
         catvalues <- c(catvalues,NA)
         if(is.na(DIVISION) | DIVISION %in% catvalues){
           nodeid <- 60
           predict <- 6973.48098027
         } else {
           catvalues <- c("NA","S23A","S48B","S49D")
           catvalues <- c(catvalues,NA)
           if(is.na(PSU) | PSU %in% catvalues){
             if(!is.na(ALCBEVPQ) & ALCBEVPQ <= 255.000000000 ){
               if(!is.na(EDUC_REF) & EDUC_REF <= 14.5000000000 ){
                 nodeid <- 488
                 predict <- 74.8427668996
               } else {
                 if(!is.na(HOUSOPCQ) & HOUSOPCQ <= 91.5000000000 ){
                   nodeid <- 978
                   predict <- 177.653541465
                 } else {
                   nodeid <- 979
                   predict <- 517.446577977
                 }
               }
             } else {
               nodeid <- 245
               predict <- 1515.68830231
             }
           } else {
             if(!is.na(PROPTXCQ) & PROPTXCQ <= 379.166650000 ){
               nodeid <- 246
               predict <- 59.4689905210
             } else {
               nodeid <- 247
               predict <- 109.668815824
             }
           }
         }
       } else {
         nodeid <- 31
         predict <- 1896.56826630
       }
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
    AGE2 <- as.numeric(newdata$AGE2[i])
    BATHRMQ <- as.numeric(newdata$BATHRMQ[i])
    CUTENURE <- as.character(newdata$CUTENURE[i])
    EARNCOMP <- as.character(newdata$EARNCOMP[i])
    EDUC_REF <- as.numeric(newdata$EDUC_REF[i])
    EDUCA2 <- as.numeric(newdata$EDUCA2[i])
    FINCBTAX <- as.numeric(newdata$FINCBTAX[i])
    FJSSDEDX <- as.numeric(newdata$FJSSDEDX[i])
    FSALARYX <- as.numeric(newdata$FSALARYX[i])
    HLFBATHQ <- as.numeric(newdata$HLFBATHQ[i])
    INC_RANK <- as.numeric(newdata$INC_RANK[i])
    NUM_AUTO <- as.numeric(newdata$NUM_AUTO[i])
    OCCUCOD1 <- as.character(newdata$OCCUCOD1[i])
    OCCUCOD2 <- as.character(newdata$OCCUCOD2[i])
    PERSOT64 <- as.numeric(newdata$PERSOT64[i])
    POPSIZE <- as.numeric(newdata$POPSIZE[i])
    RACE2 <- as.character(newdata$RACE2[i])
    REF_RACE <- as.character(newdata$REF_RACE[i])
    REGION <- as.character(newdata$REGION[i])
    VEHQ <- as.numeric(newdata$VEHQ[i])
    FOODCQ <- as.numeric(newdata$FOODCQ[i])
    ALCBEVPQ <- as.numeric(newdata$ALCBEVPQ[i])
    SHELTCQ <- as.numeric(newdata$SHELTCQ[i])
    MRTINTPQ <- as.numeric(newdata$MRTINTPQ[i])
    PROPTXPQ <- as.numeric(newdata$PROPTXPQ[i])
    PROPTXCQ <- as.numeric(newdata$PROPTXCQ[i])
    ELCTRCPQ <- as.numeric(newdata$ELCTRCPQ[i])
    WATRPSPQ <- as.numeric(newdata$WATRPSPQ[i])
    HOUSOPCQ <- as.numeric(newdata$HOUSOPCQ[i])
    PERINSPQ <- as.numeric(newdata$PERINSPQ[i])
    LIFINSPQ <- as.numeric(newdata$LIFINSPQ[i])
    RETPENCQ <- as.numeric(newdata$RETPENCQ[i])
    STATE <- as.character(newdata$STATE[i])
    ETOTALC <- as.numeric(newdata$ETOTALC[i])
    ESHELTRC <- as.numeric(newdata$ESHELTRC[i])
    EOWNDWLC <- as.numeric(newdata$EOWNDWLC[i])
    UNISTRQ <- as.character(newdata$UNISTRQ[i])
    INCLASS2 <- as.numeric(newdata$INCLASS2[i])
    FRRETIRM <- as.numeric(newdata$FRRETIRM[i])
    PSU <- as.character(newdata$PSU[i])
    HIGH_EDU <- as.numeric(newdata$HIGH_EDU[i])
    BUILT <- as.numeric(newdata$BUILT[i])
    LIQUIDX <- as.numeric(newdata$LIQUIDX[i])
    RETSURVX <- as.numeric(newdata$RETSURVX[i])
    RETSURV <- as.character(newdata$RETSURV[i])
    STOCKX <- as.numeric(newdata$STOCKX[i])
    DIVISION <- as.character(newdata$DIVISION[i])
    REFGEN <- as.numeric(newdata$REFGEN[i])
    FFTAXOWE <- as.numeric(newdata$FFTAXOWE[i])
    tmp <- predicted()
    node <- c(node,as.numeric(tmp[1]))
    pred <- c(pred,tmp[2])
}
