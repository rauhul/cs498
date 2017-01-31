setwd('~/Developer/cs498/homework1')
wdat<-read.csv('pima-indians-diabetes.txt', header=FALSE)
library(klaR)
library(caret)

# split the input data into feature vectors and associated label
x_vec<-wdat[,-c(9)]
y_vec<-wdat[,9]


# remove 0 entries from feature vectors and replace them with NA (optional operation)
#for (i in c(3, 5, 6, 8)) {
#  zeros<-x_vec[, i]==0
#  x_vec[zeros, i]=NA
#}

# allocate space for training/testing results
train_score<-array(dim=10)
test_score<-array(dim=10)

# run training/testing 10 times
for (iter in 1:10) {
  
  ## training
  
  # get indicies to use for training
  wtd<-createDataPartition(y=y_vec, p=.8, list=FALSE)
  
  # extract *training* feature vectors and labels
  x_vec_train<-x_vec[wtd, ]
  y_vec_train<-y_vec[wtd]
  
  # split into postive and negative *training* sets
  pos_train_example_flag<-y_vec_train>0
  
  pos_train_examples<-x_vec_train[pos_train_example_flag, ]
  neg_train_examples<-x_vec_train[!pos_train_example_flag,]
  
  # extract *testing* feature vectors and labels
  x_vec_test<-x_vec[-wtd, ]
  y_vec_test<-y_vec[-wtd]
  
  # solve for positive mean and std.dev
  pos_mean<-sapply(pos_train_examples, mean, na.rm=TRUE)
  pos_sd<-sapply(pos_train_examples, sd, na.rm=TRUE)

  # solve for negative mean and std.dev
  neg_mean<-sapply(neg_train_examples, mean, na.rm=TRUE)
  neg_sd<-sapply(neg_train_examples, sd, na.rm=TRUE)
  
  ## evaluating - training data
  
  # Solve for log probability that each feature vector corresponds to a positive label
  pos_train_offset<-t(t(x_vec_train)-pos_mean)
  pos_train_scaled<-t(t(pos_train_offset)/pos_sd)
  pos_train_log_prob<--(1/2)*rowSums(apply(pos_train_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(pos_sd))
 
  # Solve for log probability that each feature vector corresponds to a negative label
  neg_train_offset<-t(t(x_vec_train)-neg_mean)
  neg_train_scaled<-t(t(neg_train_offset)/neg_sd)
  neg_train_log_prob<--(1/2)*rowSums(apply(neg_train_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(neg_sd))
  
  # record percentage guessed correctly in scores array
  guesses_train<-pos_train_log_prob>neg_train_log_prob
  num_correct_train<-guesses_train==y_vec_train
  train_score[iter]<-sum(num_correct_train)/(sum(num_correct_train)+sum(!num_correct_train))
  
  ## evaluating - testing data
  
  # Solve for log probability that each feature vector corresponds to a positive label
  pos_test_offset<-t(t(x_vec_test)-pos_mean)
  pos_test_scaled<-t(t(pos_test_offset)/pos_sd)
  pos_test_log_prob<--(1/2)*rowSums(apply(pos_test_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(pos_sd))
  
  # Solve for log probability that each feature vector corresponds to a negative label
  neg_test_offset<-t(t(x_vec_test)-neg_mean)
  neg_test_scaled<-t(t(neg_test_offset)/neg_sd)
  neg_test_log_prob<--(1/2)*rowSums(apply(neg_test_scaled, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(neg_sd))
  
  # record percentage guessed correctly in scores array
  guesses_test<-pos_test_log_prob>neg_test_log_prob
  num_correct_test<-guesses_test==y_vec_test
  test_score[iter]<-sum(num_correct_test)/(sum(num_correct_test)+sum(!num_correct_test))
}