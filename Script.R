library(inTrees)
library('dplyr')
library('Matrix')
shrooms = read.csv('mushrooms.csv')
shrooms = na.omit(shrooms)

# Our output is the class of the mushroom. Either edible (e) or poisonous (p)
# Converting the class to characters
shrooms['class'] %>% mutate_if(is.factor, as.character) -> shrooms['class']
# Converting the classes ('p'/'e') to numeric values
a= 0
for (i in unique(shrooms$class)) {
  shrooms$class[shrooms$class == i] = a
  a = a+1
}

set.seed(42)

# Splitting into train and test datasets (0.75 ratio)

sample = sample.split(shrooms, SplitRatio = 0.75)
train1 = subset(shrooms,sample==TRUE)
test1 = subset(shrooms,sample==FALSE)


output_vector = train1[,class]

# For future accuracy testing
real_output = test1[,class]

# Our model is classifying into two classes (poisonous or edible)
number_classes = 2

# Making the model
xgbmodel = xgboost(data = data.matrix(train1[,-1]), label = output_vector,nfold = 50,max.depth = 4,nrounds = 15,eta = 1,nthread = 4, objective = 'multi:softprob', num_class = number_classes)

# Testing accuracy
pred <- predict(xgbmodel, data.matrix(test1[,-1]))
# Rounding because we need 0 or 1 values
pred <- lapply(pred, round)
# Wherever the predictions is 1
pred = pred==1
print("Accuracy")
print(sum(pred==real_output)/length(real_output)*100)


# Now we know that our model is performing well, but if someone wants to know why a mushroom is poisonous, what do we say
# We can use feature importance, and also use inTrees to see the rules that are defining the tree ensemble

# Finding feature importance
importance <- xgb.importance(feature_names = colnames(data.matrix(train1[,-1])), model = xgbmodel)
print(head(importance))

# Using inTrees:
# Making a tree list from the XGB model:
tree_list = XGB2List(xgbmodel, data.matrix(train1[,-1]))
# Extracting rules from the tree list:
exec = extractRules(tree_list, data.matrix(train1[,-1]))
# Getting rule metrics from the rules:
ruleMetric <- getRuleMetric(exec,data.matrix(train1[,-1]),output_vector)
# print(ruleMetrics)
# Rule Metric explanation:
# len: No. of variable-value pairs in that condition
# freq: %age of data satisfying the condition 
# pred: Outcome
# err: Error rate




