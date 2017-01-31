setwd('~/Developer/cs498/homework1')
wdat<-read.csv('processed.cleveland.txt', header=FALSE)
library(klaR)
library(caret)

# Open the data file
all_data<-read.csv('processed.cleveland.txt', header = FALSE)

# split the input data into feature vectors and associated label
x_vec<-wdat[,-c(dim(wdat)[2])]
y_vec<-as.factor(wdat[,dim(wdat)[2]])

# get indicies to use for training
wtd<-createDataPartition(y = y_vec, p = .85, list = FALSE)

# extract training feature vectors and labels
x_vec_train<-x_vec[wtd, ]
y_vec_train<-y_vec[wtd]

# extract testing feature vectors and labels
x_vec_test<-x_vec[-wtd, ]
y_vec_test<-y_vec[-wtd]

# train the model with naive bayes
model<-train(x_vec_train, y_vec_train, 'nb', trControl = trainControl(method='cv', number = 10))

# test the model using the data we did not use for training
teclasses<-predict(model, newdata = x_vec_test)

# diplsy confusion matrix of results
confusionMatrix(data = teclasses, y_vec_test)

# print accuracy statistics
mean(model$resample$Accuracy)
sd(model$resample$Accuracy)
