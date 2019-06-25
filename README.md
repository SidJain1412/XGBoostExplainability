# XGBoostExplainability

#### An XGBoost (or any other ensemble model) is a black box, and one can't decipher why the predictions are what they are.
(view the **Notebook.md** file for the notebook, .Rmd files for the source code)
#### In this R notebook I demonstrate how to use the inTrees library to interpret and uncover the basic rules influencing the decision trees.
(or atleast my understanding of it)

Flow of inTrees:

![Flow Diagram](https://github.com/SidJain1412/XGBoostExplainability/blob/master/Model%20Explainability%20Flow.jpg "Flow Diagram")


[Reference Paper](https://arxiv.org/abs/1408.5456 "Arxiv Link to Paper")

[Documentation](https://rdrr.io/cran/inTrees/ "Rdrr link")

Using the Mushroom dataset, classifying mushrooms into poisonous or edible based on various features ([Dataset Link](https://www.kaggle.com/uciml/mushroom-classification))
