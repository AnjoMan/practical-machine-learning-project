#' Project - Practical Machine Learning - Anton Lodder
#' February 21st, 2015
#'
#'
#' First we load the library and set the seed to ensure repeatability.
library(caret)
set.seed(3433)

#' The training data comes in a .csv file. It is necessary to catch 
#' empty cells, text 'N/A' and '#DIV/0' entries using the 
#' na.strings option; this ensures that R can ignore these values
#' rather than treating them as factors.

mData <- read.csv('pml-training.csv',na.strings = c("#DIV/0!", "", " ", "NA"));



#' Several columns contain only periodic reports of 
#' the kurtosis and skewness of specific variables. 
#' These columns are mostly empty or contain NA values
#' which we want to remove so we can build a model on just
#' the columns containing predictors. This next line
#' filters out columns that have 
#' a lot of elements containing 'na', eliminating the irrelevant columns.

mData <- mData[, which(as.numeric(colSums(is.na(mData)))/nrow(mData) <= .90)]

#' In addition to removing the mostly empty columns, 
#' it is necessary to strip out metadata (index, name, window, 
#' timestamp etc...) that aren't predictors. These are 
#' contained in the first seven columns so they are easy to remove.
mData <- mData[,-c(1:7)]


#' The training data-set contains
{{dim(mData)[1]}}
#' elements. We want to partition the data into a training and cross-validation 
#' set; the training set will be used to build our classifier and the testing
#' set will be used to estimate our out-of-sample error.
#' 
#' Since there is a large number of samples, sample bias is less of a 
#' concern for this problem. We can get a better estimate of the out-of-sample
#' error by using a larger cross-validation set. I chose to split the training 
#' and cross-validation data into equal portions by setting p=0.5.

inTrain <- createDataPartition(y=mData$classe, p=0.5, list=FALSE)

training <- mData[inTrain,]
cross_validate <- mData[-inTrain,]

#' We can run
table(training[,53])
#' to get the sample tallies for each class. There are 1500+ 
#' samples for each class, which means we have a large enough 
#' training set that we can hopefully get a properly generalized
#' classifier *and* a large enough cross-validation set to ensure
#' a good out-of-sample error estimate.
#'
#' I've chosen to use the random forest model, which can be built as follows:

library(randomForest)
modFit <- randomForest(training$classe ~., data=training)

#' The final model looks like this:
modFit

#' The outcome of the fitting process gives an out-of-bag (OOB) 
#' or out-of-sample error estimate of 0.87%. This quite low, 
#' which is what we want.
#' With random forests we don't need to run cross-validation to 
#' reduce the variability of the error, since this process is 
#' embedded in the structure of a random forest classifier.
#'
#' When we run prediction on the cross-validation data, we end up
#' with the following accuracy estimates for each class: 
pred <- predict(modFit, newdata=cross_validate)

confusionMatrix(cross_validate$classe, pred)
#' All of the accuracy measures are above 99% and the average accuracy is 99.2%. This shows that we
#' have obtained a good performance from our model.
#'
#' We can plot the variable importance values to get an idea
#' of which predictors have been emphasized by the random 
#' forest and which have been de-emphasized.

varImpPlot(modFit)

#' This shows that the most relevant variables selected by 
#' our classifier are _roll_belt_ and _yaw_belt_, followed by
#' a number of other variables. After the first 7 variables,
#' the importance of subsequent variables tails off slowly, 
#' indicating that the information in the data-set is 
#' distributed fairly evenly accross the data-set.
#'
#' If we use only the first 7 predictors, we can see what the 
#' we are gaining from having the extra variables:
modFit2 <- randomForest(classe ~ roll_belt+yaw_belt+pitch_forearm+magnet_dumbbell_z+pitch_belt+magnet_dumbbell_y+roll_forearm,data=training)
modFit2

pred2 <-predict(modFit2, newdata=cross_validate)
conf_2<-confusionMatrix(cross_validate$classe, pred2)
conf_2$overall

#' This shows that even with only the top 7 variables
#' it is possible to get an accurate classifier; however,
#' the jump from 98% to 99.2% is not insignificant, 
#' representing a halving of the error rate.
#'
#' We can look at partial dependance plots to get a rough
#' idea of how classification would change for a particular
#' variable.

partialPlot(modFit, training[,-53], x.var='roll_belt')

partialPlot(modFit, training[,-53], x.var='roll_forearm')

partialPlot(modFit, training[,-53], x.var='roll_dumbbell')


#'
#' Finally, we can run the model on the 
#' test data to predict what class of 
#' exercise each data point might be. It is necessary to run the same 
#' transformations on the test data in order to ensure that the classifier
#' can be properly applied.

test_set <- read.csv('pml-testing.csv',na.strings = c("#DIV/0!", "", " ", "NA"));
test_set <- test_set[, which(as.numeric(colSums(is.na(test_set)))/nrow(test_set) <= .9)]
testing <- test_set[,-c(1:7)]

pred_test <- predict(modFit, newdata=testing)

#' The outcomes on the test set are:
pred_test

