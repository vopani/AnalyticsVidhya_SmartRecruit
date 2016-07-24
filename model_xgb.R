## setting working directory and seed
path <- "./AnalyticsVidhya/SmartRecruit"
setwd(path)

seed <- 235
set.seed(seed)


## loading libraries
library(data.table)
library(xgboost)


## loading data
X_train <- fread("./Train_pjb2QcD.csv")
X_test <- fread("./Test_wyCirpO.csv")


## feature engineering
X_train <- subset(X_train, !Manager_Joining_Designation %in% c("Level 7", "Other"))

# cleaning raw features
X_train <- X_train[, ":="(ID = NULL,
                          Office_PIN = as.numeric(Office_PIN),
                          Application_Receipt_Date = as.numeric(as.Date("2016-01-01") - as.Date(Application_Receipt_Date, "%m/%d/%Y")),
                          Applicant_City_PIN = as.numeric(Applicant_City_PIN),
                          Applicant_Gender = ifelse(Applicant_Gender == "M", -1, ifelse(Applicant_Gender == "F", 1, 0)),
                          Applicant_Age = as.numeric(as.Date("2016-01-01") - as.Date(Applicant_BirthDate, "%m/%d/%Y"))/365,
                          Applicant_BirthDate = NULL,
                          Applicant_Marital_Status = NULL,
                          Applicant_Occupation = as.numeric(as.factor(Applicant_Occupation)),
                          Applicant_Qualification = NULL,
                          Manager_Experience = as.numeric(as.Date("2016-01-01") - as.Date(Manager_DOJ, "%m/%d/%Y"))/365,
                          Manager_DOJ = NULL,
                          Manager_Joining_Designation = as.numeric(as.factor(Manager_Joining_Designation)),
                          Manager_Current_Designation = as.numeric(as.factor(Manager_Current_Designation)),
                          Manager_Grade = NULL,
                          Manager_Status = NULL,
                          Manager_Gender = NULL,
                          Manager_Age = as.numeric(as.Date("2016-01-01") - as.Date(Manager_DoB, "%m/%d/%Y"))/365,
                          Manager_DoB = NULL,
                          Manager_Num_Application = as.numeric(Manager_Num_Application),
                          Manager_Num_Coded = as.numeric(Manager_Num_Coded),
                          Manager_Business = as.numeric(Manager_Business),
                          Manager_Num_Products = as.numeric(Manager_Num_Products),
                          Manager_Business2 = NULL,
                          Manager_Num_Products2 = NULL)]

X_test <- X_test[, ":="(Office_PIN = as.numeric(Office_PIN),
                        Application_Receipt_Date = as.numeric(as.Date("2016-01-01") - as.Date(Application_Receipt_Date, "%m/%d/%Y")),
                        Applicant_City_PIN = as.numeric(Applicant_City_PIN),
                        Applicant_Gender = ifelse(Applicant_Gender == "M", -1, ifelse(Applicant_Gender == "F", 1, 0)),
                        Applicant_Age = as.numeric(as.Date("2016-01-01") - as.Date(Applicant_BirthDate, "%m/%d/%Y"))/365,
                        Applicant_BirthDate = NULL,
                        Applicant_Marital_Status = NULL,
                        Applicant_Occupation = as.numeric(as.factor(Applicant_Occupation)),
                        Applicant_Qualification = NULL,
                        Manager_Experience = as.numeric(as.Date("2016-01-01") - as.Date(Manager_DOJ, "%m/%d/%Y"))/365,
                        Manager_DOJ = NULL,
                        Manager_Joining_Designation = as.numeric(as.factor(Manager_Joining_Designation)),
                        Manager_Current_Designation = as.numeric(as.factor(Manager_Current_Designation)),
                        Manager_Grade = NULL,
                        Manager_Status = NULL,
                        Manager_Gender = NULL,
                        Manager_Age = as.numeric(as.Date("2016-01-01") - as.Date(Manager_DoB, "%m/%d/%Y"))/365,
                        Manager_DoB = NULL,
                        Manager_Num_Application = as.numeric(Manager_Num_Application),
                        Manager_Num_Coded = as.numeric(Manager_Num_Coded),
                        Manager_Business = as.numeric(Manager_Business),
                        Manager_Num_Products = as.numeric(Manager_Num_Products),
                        Manager_Business2 = NULL,
                        Manager_Num_Products2 = NULL)]

# order of applicants
X_train$Order <- seq(0, nrow(X_train)-1)
X_test$Order <- seq(0, nrow(X_test)-1)

X_train_order <- X_train[, .(Max_Order = max(Order),
                             Min_Order = min(Order)), .(Application_Receipt_Date)]
X_test_order <- X_test[, .(Max_Order = max(Order),
                           Min_Order = min(Order)), .(Application_Receipt_Date)]

X_train <- merge(X_train, X_train_order, by="Application_Receipt_Date")
X_test <- merge(X_test, X_test_order, by="Application_Receipt_Date")

# extracting target and ids
X_target <- as.numeric(X_train$Business_Sourced)
X_ids <- X_test$ID

# normalizing order to [0,1]
X_train <- X_train[, ":="(Order_Percentile = (Order - Min_Order) / (Max_Order - Min_Order),
                          Order = NULL,
                          Max_Order = NULL,
                          Min_Order = NULL,
                          Application_Receipt_Date = NULL,
                          Business_Sourced = NULL)]

X_test <- X_test[, ":="(Order_Percentile = (Order - Min_Order) / (Max_Order - Min_Order),
                        Order = NULL,
                        Max_Order = NULL,
                        Min_Order = NULL,
                        Application_Receipt_Date = NULL,
                        ID = NULL)]

# preparing train and test data
X_test <- X_test[, .SDcols=names(X_train)]

X_train[is.na(X_train)] <- -1
X_test[is.na(X_test)] <- -1


## xgboost
model_xgb_cv <- xgb.cv(data=as.matrix(X_train), label=as.matrix(X_target), nfold=10, objective="binary:logistic", nrounds=200, eta=0.05, max_depth=6, subsample=0.75, colsample_bytree=0.8, min_child_weight=1, eval_metric="auc")
model_xgb <- xgboost(data=as.matrix(X_train), label=as.matrix(X_target), objective="binary:logistic", nrounds=200, eta=0.05, max_depth=6, subsample=0.75, colsample_bytree=0.8, min_child_weight=1, eval_metric="auc")

# CV: 0.8872
# LB: 0.8856

# variable-importance of xgb

#                    Feature        Gain      Cover     Frequence
#1:            Order_Percentile 0.623868755 0.27400047 0.15058461
#2:               Applicant_Age 0.061201384 0.12953506 0.14223307
#3:          Applicant_City_PIN 0.051893086 0.10410755 0.10420146
#4:            Manager_Business 0.048100187 0.08464180 0.10458692
#5:          Manager_Experience 0.047129010 0.08703243 0.10317358
#6:                 Manager_Age 0.043711636 0.08667838 0.10497238
#7:                  Office_PIN 0.036538730 0.08606629 0.07850443
#8:        Manager_Num_Products 0.023397802 0.03542034 0.05409225
#9:     Manager_Num_Application 0.020339642 0.02653504 0.05383528
#10:        Applicant_Occupation 0.012618119 0.02512115 0.03392008
#11:           Manager_Num_Coded 0.011136470 0.01716891 0.02633946
#12: Manager_Current_Designation 0.007966100 0.01444329 0.01811641
#13: Manager_Joining_Designation 0.006769607 0.01404605 0.01400488
#14:            Applicant_Gender 0.005329473 0.01520324 0.01143518

# prediction
pred <- predict(model_xgb, as.matrix(X_test))


## submission
submit <- data.frame("ID"=X_ids, "Business_Sourced"=pred)
write.csv(submit, "./submit.csv", row.names=F)
