---
title: "Which Class(e) do you belong to??"
author: "Ryan Pines"
output: html_document
---

## Introduction and Purpose

The goal of this assignment is to gather and analyze the data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants in order to predict the manner in which they did the exercise (this will correspond to the "classe" variable in the training set).

For this assignment, we have been provided with a training and test data set. We will gather and clean the data from both data sets, as well as perform cross validations and build models on our training data set. We will use the best fit model that we build on our test data set. Specifically, we will use our model to predict 20 different test cases.

Below are the URLs for the Training and Test Data Sets:

**Training:** "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"  
**Testing:** "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"  

## Establishing the Working Directory and Loading Libraries

```r
filePath <- "C:\\Users\\574996\\Desktop\\R_Assignments\\Machine Learning"
setwd(filePath)
```


```r
library(dplyr)
library(caret)
library(rpart)
library(randomForest)
library(gbm)
library(plyr)
```

## Data Gathering
The below code describes how the data was downloaded. This is a one time process, and as a result, the code is commented out

```r
## trainFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
## testFileURL <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

## download.file(trainFileURL, destfile = "./Machine Learning/Train_Dataset.csv", mode = "wb")
## download.file(testFileURL, destfile = "./Machine Learning/Test_Dataset.csv", mode = "wb")
```

Read in the "Train" and "Test" Data Set CSV Files. On both files, the null strings are denoted by the following: **""**, **NA**, and **#DIV/0!**

```r
trainData <- tbl_df(read.csv(file.path(filePath, "Train_Dataset.csv"), header = TRUE, na.strings = c("","NA","#DIV/0!")))
testData <- tbl_df(read.csv(file.path(filePath, "Test_Dataset.csv"), header = TRUE, na.strings = c("","NA","#DIV/0!")))
```

## Cross-Validation and Partitioning
Given that we have a large data set, we ideally want 60% of the data to be training, 20% of the data to be testing, and 20% of the data to be validation. Given that there is no validation data set, 75% of the data will be training and 25% of the data will be testing (the ratio of 60:20 is equal to the ratio of 75:25)

For cross validation on our training data set, we will perform k-fold cross validation, which partitions the training data set into k equal subsets. In this case, we will let k be 4. 3 of the subsets will be training data for which we build our model, and 1 of the subsets will be testing data for which test our model on.

Our first step will be to set the seed for purposes of reproducibility

```r
set.seed(159246837)
```

Now we perform our K-fold Cross Validation

```r
numFolds <- 4

folds <- rep_len(1:numFolds, nrow(trainData))
folds <- sample(folds, nrow(trainData))

for(k in 1:numFolds) {
    fold <- which(folds == k)
    training <- trainData[-fold,]
    testing <- trainData[fold,]
}
```

## Data Cleaning
Before we perform cross validations and build models on our training data set, we will clean up both our training and test data sets. Specifically, we will perform the following 3 actions:  

(1) Remove all variables/columns with near zero variance  
(2) Remove all columns with a high percentage of null values (> 95%)
(3) Remove all other columns that are not necessary for building our model

### (1) Removing all variables/columns with near zero variance

```r
nsvTrain <- nearZeroVar(training, saveMetrics = TRUE)
training <- training[,nsvTrain$zeroVar == FALSE || nsvTrain$nzv == FALSE]

nsvTest <- nearZeroVar(testing, saveMetrics = TRUE)
testing <- testing[,nsvTest$zeroVar == FALSE || nsvTest$nzv == FALSE]
```

### (2) Remove all columns with a high percentage of null values (> 95%)
**The reason why we need to remove null values is because prediction algorithms often fail when using missing data (null values) in data sets (see Week 2 Lecture "Basic Preprocessing" for your reference).** The reason why we choose 95% instead of 100% is that some variables may be extremely important for our prediction, yet may have some missing values

First, we will assign the number that corresponds to 95% of the rows for both the training and test sets to 2 different objects

```r
Train95 <- 0.95 * nrow(training)
Test95 <- 0.95 * nrow(testing)
```

Afterwards, we will factor out all variables/column that have more than 95% of null values.

```r
training <- training[,colSums(is.na(training)) < Train95]
testing <- testing[,colSums(is.na(testing)) < Test95]
```

### (3) Remove all other columns that are not necessary for building our model  
From analyzing the both the Train Dataset and Test Dataset CSV files, the following variables appear to be identifier variables, and are not necessary for our model: **Column1**,**user_name**,**cvtd_time**. These variables correspond to the 1st, 2nd, and 5th columns respectively in both CSV files

As a result, we will remove these variables

```r
training <- training[,-c(1,2,5)]
testing <- testing[,-c(1,2,5)]
```

Verify the rows and columns in the the training and testing data sets. Both data sets should have the same number of columns

```r
dim(training)
```

```
## [1] 14717    57
```

```r
dim(testing) 
```

```
## [1] 4905   57
```


## Building Models
We will build prediction models with our partitioned training data set by using the following methods: 

### Classification Tree
Create the Classification Tree Model

```r
set.seed(546372819)
modelFitCT <- train(classe ~ ., data = training, method = "rpart")
```

### Random Forests
Create the Random Forest Model

```r
set.seed(546372819)
modelFitRF <- randomForest(classe ~., data = training, method = "class")
```

## Predicting with the Models and Analyzing Predictions
Now that we built our models, we will use the partitioned testing data set for making our predictions. After making our predictions, we will generate a confusion matrix for our predictions. The Confusion Matrix will allow us to analyze the **accuracy**, as well as **expected out of sample error**, for both of our model's predictions.

### Making the predictions

```r
predictionCT <- predict(modelFitCT, testing) ## Classification Tree
predictionRF <- predict(modelFitRF, testing) ## Random Forest
```

### Confusion Matrices and Analyses for the predictions

```r
confusionMatrix(predictionCT, testing$classe) ## Classification Tree
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1230  261  143  127   32
##          B   22  322   29  157   77
##          C  136  385  679  489  247
##          D    0    0    0    0    0
##          E    2    0    0   38  529
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5627          
##                  95% CI : (0.5487, 0.5766)
##     No Information Rate : 0.2834          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4412          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8849  0.33264   0.7979   0.0000   0.5977
## Specificity            0.8398  0.92761   0.6899   1.0000   0.9900
## Pos Pred Value         0.6860  0.53048   0.3507      NaN   0.9297
## Neg Pred Value         0.9486  0.84970   0.9421   0.8347   0.9179
## Prevalence             0.2834  0.19735   0.1735   0.1653   0.1804
## Detection Rate         0.2508  0.06565   0.1384   0.0000   0.1078
## Detection Prevalence   0.3655  0.12375   0.3947   0.0000   0.1160
## Balanced Accuracy      0.8624  0.63013   0.7439   0.5000   0.7939
```

```r
confusionMatrix(predictionRF, testing$classe) ## Random Forest
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1390    0    0    0    0
##          B    0  968    4    0    0
##          C    0    0  845    0    0
##          D    0    0    2  810    2
##          E    0    0    0    1  883
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9982          
##                  95% CI : (0.9965, 0.9992)
##     No Information Rate : 0.2834          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9977          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   0.9929   0.9988   0.9977
## Specificity            1.0000   0.9990   1.0000   0.9990   0.9998
## Pos Pred Value         1.0000   0.9959   1.0000   0.9951   0.9989
## Neg Pred Value         1.0000   1.0000   0.9985   0.9998   0.9995
## Prevalence             0.2834   0.1973   0.1735   0.1653   0.1804
## Detection Rate         0.2834   0.1973   0.1723   0.1651   0.1800
## Detection Prevalence   0.2834   0.1982   0.1723   0.1660   0.1802
## Balanced Accuracy      1.0000   0.9995   0.9965   0.9989   0.9987
```
From the above Confusion Matrices, we see that the Accuracy for the Classification Tree model is around 56.27%. As a result, the expected out of sample error for the Classification Tree model is 43.73%, which is fairly high.

Whereas, we see that the Accuracy for the Random Forest model is around 99.82%. As a result, the expected out of sample error for the Random Forest model is 0.18%, which is extremely low.

## Conclusions
Given that the accuracy for the Random Forest model is around 99.82%, this is the model that we will use for our predictions. From the Confusion Matrix, there were only 9 incorrect predictions (out of 4907 total predictions). The number of incorrect predictions is calculated by summing the nondiagonal terms on the matrix. Given that there were so little incorrect predictions, this would explain why the expected out of sample error rate for the Random Forest model is so low, and as a result, would also explain why the Random Forest Model is the ideal model to use for making predictions.
