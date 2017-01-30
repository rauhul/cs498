setwd('~/Developer/cs498/homework1')
wdat<-read.csv('pima-indians-diabetes.txt', header=FALSE)
library(klaR)
library(caret)

# split the input data into feature vectors and associated label
x_vec<-wdat[,-c(9)]
y_vec<-as.factor(wdat[,9])

# get indicies to use for training
wtd<-createDataPartition(y=y_vec, p=.8, list=FALSE)

# extract training feature vectors and labels
x_vec_train<-x_vec[wtd, ]
y_vec_train<-y_vec[wtd]

# extract testing feature vectors and labels
x_vec_test<-x_vec[-wtd, ]
y_vec_test<-y_vec[-wtd]

# train the model with a SVM
svm<-svmlight(x_vec_train, y_vec_train, pathsvm='~/Developer/cs498/svm_light_OS10.8.4_i7')

# use model to classify the data we did not use for training
labels<-predict(svm, x_vec_test)

# compute percent correct classifications
guess<-labels$class
sum(guess==y_vec_test)/(sum(guess==y_vec_test)+sum(!(guess==y_vec_test)))