library(data.table)
library(tidyverse)
library(inspectdf)
library(caret)
library(h2o)
library(rstudioapi)
library(highcharter)


#Import Data

my_data <- fread("mushrooms.csv")


h2o.init()

h2o_data <- my_data %>% as.h2o()


h2o_data <- h2o_data %>% h2o.splitFrame(ratios = 0.8, seed = 123)
train <- h2o_data[[1]]
test <- h2o_data[[2]]

target <- 'class'
features <- my_data %>% select(-class) %>% names()

model <- h2o.automl(
  x = features, y = target,
  training_frame = train,
  validation_frame = test,
  leaderboard_frame = test,
  stopping_metric = "AUC",
  balance_classes = T,
  nfolds = 10, seed = 123,
  max_runtime_secs = 480)


model@leaderboard %>% as.data.frame()
model@leader 

# Predicting the Test set results
pred <- model@leader %>% h2o.predict(test) %>% as.data.frame()


# 3. Find threshold by max F1 score

model@leader %>% 
  h2o.performance(test) %>% 
  h2o.find_threshold_by_max_metric('f1') -> treshold


# 4. Calculate Accuracy, AUC, GINI.

# Confusion Matrix
confmat <- model@leader %>% 
  h2o.confusionMatrix(test) %>% 
  as_tibble() %>% 
  select("0","1") %>% 
  .[1:2,] %>% t()

# Accuracy

paste("Accuracy = ",round(sum(diag(confmat))/sum(confmat)*100,1),"%")

accuracy <- model@leader %>% h2o.performance(test) %>% h2o.accuracy()

# AUC

auc <- model@leader %>% h2o.performance(test) %>% h2o.auc()


# GINI
gini <- model@leader %>% h2o.performance(test) %>% h2o.giniCoef()


