# Ensemble models for classification (combine deep learning with machine learning)

**Objetive:**

To develop a robust approach to conduct classification on data (a person is wearing glasses or not) using a ensemble of models, 
which include machine learning models (random forest,Gradient Boosting and Extra Trees) and deep learning model (optimized NN using Bayesian optimization).

**Data:**

The data for this project can be downloaded from [Kaggle](https://www.kaggle.com/c/applications-of-deep-learningwustl-spring-2020).

Specificall, there are two data were used for this project:

 **(1)** training.csv, which include the 512 features one response variable glasses (1 represent have glass, 0 means no glasses).
 
 **(2)** submit.csv, which in fact is the test data which to measure how good of the model. (not really used in this project, but it's useful
 for the Kaggle practice)
 
 I used the the same dataset of the Bayesian optimization deep learning repo. If you are interested in, you can go to [here](https://github.com/tankwin08/Bayesian_optimization_deep_learning) to check.

**[Ensemble Modeling](https://www.sciencedirect.com/topics/computer-science/ensemble-modeling):**

Ensemble modeling is a process where multiple diverse models are created to predict an outcome, either by using many different modeling algorithms or using different 
training data sets. The ensemble model then aggregates the prediction of each base model and results in once final prediction for the unseen data. 
The motivation for using ensemble models is to reduce the generalization error of the prediction. As long as the base models are diverse and independent, 
the prediction error of the model decreases when the ensemble approach is used. 

The approach seeks the wisdom of crowds in making a prediction. Even though the ensemble model has multiple base models within the model, 
it acts and performs as a single model. 

**Why Ensemble modeling?**

There are two major benefits of Ensemble models:

	Better prediction

	More stable model

The aggregate opinion of a multiple models is less noisy than other models. 
In finance, it was called “Diversification”,  a mixed portfolio of many stocks will be much less variable than just one of the stocks alone. 

This is also why your models will be better with ensemble of models rather than individual. 
One of the caution with ensemble models are over fitting although bagging takes care of it largely.


**Cross validation vs. [Bootstrap](https://stats.stackexchange.com/questions/18348/differences-between-cross-validation-and-bootstrapping-to-estimate-the-predictio)**

1 Bootstrapping is a way to quantify the uncertainty in your model while cross validation is used for model selection and measuring predictive accuracy.

2 CV tends to be less biased but K-fold CV has fairly large variance. On the other hand, bootstrapping tends to drastically reduce the variance 
   but gives more biased results (they tend to be pessimistic). 
 

**In python,** 

For bootstrapping 

            ShuffleSplit - to perform the splits. 

For k-fold CV:  

			KFold - a regression problem. 
			
            StratifiedKFold - a classification problem.



**Method Overview**

**1** Base on optimized parameters such as dropout,neuronPct, neuronShrink to form the NN models.

**2** Build a model ensembler with seven models and 10 folds cross validation.

**3** For each fold, used 9/10 of training data (9 folds) to build th
e model, the rest to conduct prediction. Meanwhile, we also used the trained model to predict at the test data. Here, different strategies have been applied to collect the validation prediction results (from cv of training data) and submit prediction results. 

Regarding the **validation prediction results**, it need to collect in each fold as we only have 1/10 of training data to obtian the prediction results. As follows, the test will tell you which row index we should fill in. At the end of 10 folds, we should have every values in the one column (represent one model).
```    
                                     dataset_blend_train[test, j] = pred[:, 1]
```
Regarding the **test prediction results**, we collected the prediction results for the submit data for each fold and each model. At the end of each model, we need to do avergae of these prediction results of these 10 folds.

**4** Build a logstic regresson between y and the validation prediction results. Use this model to conduct the prediction of our submit prediction results. This results is more like to assign weights to these models for prediction of final output.

**5** Format te output and save results.




