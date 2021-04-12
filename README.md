# MNIST without Neural Networks
## Introduction

The MNIST dataset is a set of 70,000 handwritten digits. Of these 42,000 are used for training while 28,000 are used for testing our trained models. The dataset can be found on Kaggle. The MNIST is an excellent standard benchmark dataset and can be used an a playground to to test and learn more about standard classification algorithms.
Using Neural Networks it's reasonably easy to achieve an accuracy greater than 98% as is evidenced by the submitted notebooks on Kaggle. In this blog we will try to see what's the best accuracy we can get without using deep neural networks. We will use Random Forest, Gradient Boost, SVM, stacking and combinations of the above to perform our analysis. We will try different tricks and feature engineering, not all of which will give us better accuracy compared to out of box classification methods. We will use SVD to try to speed up our analysis.Not all of our experimentation will yield results. However since the goal of this project is to experiment and learn, we will report our results.


## Data Visualization and Features Engineering
The dataframe of the training data is shown below.Label indicates the digit it is. The 784 pixel columns are the intensity of each pixel from 0 to 255. We can reshape the data to a 28 X 28 image. Below are plotted random images of the 16 randomly selected digits. Our job now is to classify them.

![Alt text](/Figures/transformed_data.png?raw=true "Data")

## Models
There is no good reason why linear regression would work. For example, both too low and too high temperature would be bad for bike sharing numbers. There is also no reason why other continuous variables are related linearly. Thus we choose Random Forest and XGB as our two regression models.

The metric used to evaluate the model is Root Mean Square Log Error (RMSLE), i.e. the root mean square error of log(1 + prediction). This makes predicting both high and low counts equally important.

Since we are using RMSLE our target will always be the log of the registered, casual or total counts, depending on the target we are dealing with.

We will begin by considering the total count as our target.
### Random Forest Regression

We use sklearn RandomForestRegressor as our regression model. We use GridSearchCV to scan over a grid of parameters to find the best value. We do a (80,20) split on our training data to split between training and "test" data. We use 5-fold cross-validation to avoid overfitting. The following are our GridSearch parameters

'n_estimators': [100],
'criterion': ['mse'],
'max_depth': [None, 2, 3, 5,10],
'min_samples_split': [None,5,10],
'min_samples_leaf': [None,3,5],

The training is done under a minute. We get
test RMSLE = 0.34
submission RMSLE = 0.42

There is a big discrepancy between predicting on the test data and blind (output not known to us) submission data. It dosen't seem that our model is overfitting since training and test error are similar. The more likely case is that since the "blind" submission data is for the latter dates for every month (20th and beyond), the behavior of users at the end of the month might be different than from the beginning of the month. However we leave addressing this for future work. Note that the best submission RMSLE is 0.33, so we are still quite far away from having a great model.

## XGBoost Refression
Next we use a XGBoost regression model on our data. Other things remain the same, the GridSearchCV parameters are given by

objective= 'reg:squarederror'
'max_depth': range (2, 10, 2),
'n_estimators': range(20, 120, 20),
'learning_rate': [0.001,0.01, 0.1]

The training is again done in about a minute. We get

test RMSLE = 0.32
submission RMSLE = 0.402

### Basic "Stacked" Model
We take our best fit model from both Random Forest and XGB and linealry combine them to get our final prediction. The linear fit suggests y_test = 0.33* y_RF + 0.67*y_xgb. We get 

submission RMSLE = 0.399

which is a slight improvement over our XGB only model

### XGB separate fit
Since XGB performs reasonably well on our data, we now use it to train two separate models. Since we have casual and registered user counts, it makes sense that two separate models fit each of those categories since each category of a user might have different performances.

All the hyperparameter tuning remains the same. For our registered user model we get

test RMSLE = 0.31

For our casual user model we get

test RMSLE = 0.51

As we can see the casual data is not fit well at all and has a high variance. Combining the results we get

test RMSLE = 0.31
submission RMSLE = 0.400

which is only slightly better than our total XGBoost model.

We also try to use the registered prediction as a feature for our casual usage prediction. Though casual usage test RMSLE improves slightly it has extremely minimal impact on the final total RMSLE since the number of casual users is much less than registered users.

### Conclusion and Future Work
RF and XGB are powerful regressors since they dont require linearity and minimal feature engineering to get reasonable results. While our best performance is far from best performance on the Kaggle Leaderboard, we can attempt the following in the future
1) Train more regressors to better do  a stacked analysis
2) Minimize error by accounting for the fact that the submission data is in the later third of the month. Time Series analysis over a month might help predicting the trend for the end of month.


## Blog

https://aghalsasi-datascience.blogspot.com/2021/03/bike-sharing-demand-prediction.html

## Navigating the repository

The main analysis and presentation notebook is bike_sharing_demand.