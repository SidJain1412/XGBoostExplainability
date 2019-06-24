XGBoost Explainability using inTrees
================
Siddharth Jain
24 June 2019

### This R Notebook with code explains the usage of inTrees library using a sample dataset.

### This library helps with interpreting a Extreme Gradient Boosting model in R, by creating simplified rules.

Refer the Readme file of the repository to know more.

## Imports

``` r
library('inTrees')
library('dplyr')
library('Matrix')
library('caTools')
library('xgboost')
```

## Reading the data

``` r
shrooms = read.csv('mushrooms.csv')
```

Omitting NA (empty) rows

``` r
shrooms = na.omit(shrooms)
```

**Our output is the class of the mushroom. Either edible (e) or
poisonous (p)**

Converting the class to
characters

``` r
shrooms['class'] %>% mutate_if(is.factor, as.character) -> shrooms['class']
```

Converting the classes (‘p’/‘e’) to numeric value

``` r
a = 0
for (i in unique(shrooms$class)) {
  shrooms$class[shrooms$class == i] = a
  a = a + 1
}
```

Setting seed for our train/test split on the given dataset

``` r
set.seed(42)
```

**Splitting into train and test datasets (0.75 ratio)**

``` r
sample = sample.split(shrooms, SplitRatio = 0.75)
train1 = subset(shrooms, sample == TRUE)
test1 = subset(shrooms, sample == FALSE)
```

Our output variable is in the first column (class)

``` r
output_vector = train1[, 1]
```

For future accuracy testing

``` r
real_output = test1[, 1]
```

Our model is classifying into two classes (poisonous or edible)

``` r
number_classes = 2
```

### Making the model

``` r
xgbmodel = xgboost(
  data = data.matrix(train1[, -1]),
  label = output_vector,
  nfold = 100,
  max.depth = 4,
  nrounds = 10,
  eta = 1,
  nthread = 4,
  objective = 'multi:softprob',
  num_class = number_classes
)
```

    ## [1]  train-merror:0.030147 
    ## [2]  train-merror:0.005829 
    ## [3]  train-merror:0.000833 
    ## [4]  train-merror:0.000833 
    ## [5]  train-merror:0.000833 
    ## [6]  train-merror:0.000000 
    ## [7]  train-merror:0.000000 
    ## [8]  train-merror:0.000000 
    ## [9]  train-merror:0.000000 
    ## [10] train-merror:0.000000

### Testing Model Accuracy

**Finding model predictions**

``` r
pred <- predict(xgbmodel, data.matrix(test1[, -1]))
```

Rounding because we need 0 or 1 values (binary classification)

``` r
pred <- lapply(pred, round)
```

Finding the sum of all places where prediction is the same as the real
output, and getting accuracy by simple division

``` r
print(sum(pred == real_output) / length(real_output) * 100)
```

    ## [1] 98.67925

We get pretty high accuracy on our model.

**Let us see if the rules extracted from this model (using inTrees) give
a comparable accuracy**

Now we know that our model is performing well, but if someone wants to
know *why* a mushroom is poisonous, what do we say?

We can use feature importance, and also use inTrees to see the exact
rules that are defining the tree ensemble used by XGBoost.

#### Finding feature importance

``` r
importance <-
  xgb.importance(feature_names = colnames(data.matrix(train1[, -1])), model = xgbmodel)
print(head(importance))
```

    ##              Feature       Gain      Cover  Frequency
    ## 1:        gill.color 0.28780570 0.15421114 0.05952381
    ## 2: spore.print.color 0.19909662 0.15114041 0.13095238
    ## 3:              odor 0.19868380 0.24468389 0.28571429
    ## 4:        population 0.14206203 0.07225213 0.07142857
    ## 5:         gill.size 0.10817059 0.11596370 0.08333333
    ## 6:      gill.spacing 0.01320549 0.01994578 0.04761905

## Using inTrees

#### Making a tree list from the XGB model:

``` r
tree_list = XGB2List(xgbmodel, data.matrix(train1[, -1]))
```

**Extracting rules from the tree list:**

``` r
exec = extractRules(tree_list, data.matrix(train1[, -1]))
```

    ## 188 rules (length<=6) were extracted from the first 20 trees.

**Getting rule metrics from the
rules:**

``` r
ruleMetric <- getRuleMetric(exec, data.matrix(train1[, -1]), output_vector)
knitr::kable(head(ruleMetric), floating.environment="sidewaystable")
```

| len | freq  | err   | condition                                                      | pred |
| :-- | :---- | :---- | :------------------------------------------------------------- | :--- |
| 3   | 0.006 | 0     | X\[,9\]\<=4.5 & X\[,20\]\<=2.5 & X\[,21\]\<=4.5                | 0    |
| 4   | 0.061 | 0     | X\[,8\]\<=1.5 & X\[,9\]\<=4.5 & X\[,20\]\>2.5 & X\[,21\]\<=4.5 | 1    |
| 4   | 0.003 | 0     | X\[,8\]\>1.5 & X\[,9\]\<=4.5 & X\[,20\]\>2.5 & X\[,21\]\<=4.5  | 0    |
| 4   | 0.004 | 0.375 | X\[,9\]\<=4.5 & X\[,11\]\<=3 & X\[,13\]\<=1.5 & X\[,21\]\>4.5  | 0    |
| 4   | 0.33  | 0     | X\[,9\]\<=4.5 & X\[,11\]\<=3 & X\[,13\]\>1.5 & X\[,21\]\>4.5   | 0    |
| 3   | 0.003 | 0     | X\[,9\]\<=4.5 & X\[,11\]\>3 & X\[,21\]\>4.5                    | 1    |

Rule Metric explanation:

  - len: No. of variable-value pairs in that condition
  - freq: %age of data satisfying the condition
  - pred: Outcome
  - err: Error
rate

#### Pruning the rules generated. This removes repeated, redundant rules

``` r
ruleMetric2 <- pruneRule(ruleMetric, data.matrix(train1[, -1]), output_vector)
```

#### Selecting the important rules using a regularized random forest (refer paper for formulae)

``` r
ruleMetric3 <- selectRuleRRF(ruleMetric2, data.matrix(train1[, -1]), output_vector)
```

This step greatly reduces the number of rules by only selecting the
important
ones.

### Building a learner based on our rules. We can use this to predict values and see accuracy of our extracted rules

``` r
learner <- buildLearner(ruleMetric3, data.matrix(train1[, -1]), output_vector)
knitr::kable(learner, floating.environment="sidewaystable")
```

| len | freq               | err                | condition                     | pred |
| :-- | :----------------- | :----------------- | :---------------------------- | :--- |
| 1   | 0.0494670219853431 | 0                  | X\[,5\]\<=1.5                 | 1    |
| 2   | 0.445036642238508  | 0.0295658682634731 | X\[,8\]\<=1.5 & X\[,20\]\>2.5 | 1    |
| 1   | 0.505496335776149  | 0.0688632619439868 | X\[,1\]==X\[,1\]              | 0    |

### Using the learner to predict values:

``` r
applied <- applyLearner(learner, data.matrix(test1[, -1]))
```

**Checking the accuracy of this learner built from the rules:**

``` r
print(sum(real_output == applied) / length(real_output) * 100)
```

    ## [1] 94.90566

**We get around 95% accuracy, which is pretty good for such a small
number of rules**

Our learner doesn’t have condition names, adding those to make it human
readable (using presentRules)

``` r
Simp_Learner <- presentRules(ruleMetric3, colnames(train1[, -1]))
knitr::kable(Simp_Learner, floating.environment="sidewaystable")
```

| len | freq  | err   | condition                                                           | pred | impRRF             |
| :-- | :---- | :---- | :------------------------------------------------------------------ | :--- | :----------------- |
| 2   | 0.489 | 0.027 | gill.size\<=1.5 & spore.print.color\>2.5                            | 1    | 1                  |
| 2   | 0.486 | 0.039 | odor\>3.5 & odor\<=6.5                                              | 1    | 0.133311082662759  |
| 3   | 0.009 | 0     | gill.spacing\<=1.5 & spore.print.color\>5.5 & spore.print.color\<=7 | 0    | 0.0429142607796045 |
| 2   | 0.009 | 0     | stalk.surface.below.ring\>3.5 & ring.type\<=4.5                     | 0    | 0.0402593115954764 |
| 1   | 0.049 | 0     | odor\<=1.5                                                          | 1    | 0.024454536802662  |

**Hence we have used the inTrees library to easily understand the rules
defining a decision for an XGBoost ensemble model.**

Thank [Houtao Deng](https://www.linkedin.com/in/houtao-d-48902711/) for
the inTrees library
