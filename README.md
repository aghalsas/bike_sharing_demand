# MNIST without Neural Networks
## Introduction

The MNIST dataset is a set of 70,000 handwritten digits. Of these 42,000 are used for training while 28,000 are used for testing our trained models. The dataset can be found on Kaggle. The MNIST is an excellent standard benchmark dataset and can be used an a playground to to test and learn more about standard classification algorithms.
Using Neural Networks it's reasonably easy to achieve an accuracy greater than 98% as is evidenced by the submitted notebooks on Kaggle. In this blog we will try to see what's the best accuracy we can get without using deep neural networks. We will use Random Forest, Gradient Boost, SVM, stacking and combinations of the above to perform our analysis. We will try different tricks and feature engineering, not all of which will give us better accuracy compared to out of box classification methods. We will use SVD to try to speed up our analysis.Not all of our experimentation will yield results. However since the goal of this project is to experiment and learn, we will report our results.


## Data Visualization
The dataframe of the training data is shown below.Label indicates the digit it is. The 784 pixel columns are the intensity of each pixel from 0 to 255. We can reshape the data to a 28 X 28 image. Below are plotted random images of the 16 randomly selected digits. Our job now is to classify them.

![Alt text](/Figures/digit_plot.png?raw=true "Digits")

## Random Forest
### Out of Box Random Forest
We begin by using out of the bag random forest classifier with each pixel as our feature. We perform GridSearchCV to find the most best fit parameters. We use 5-fold cross-validation. The random forest grid search model we use is given below

Number of trees - 100 Criterion - [gini, entropy] (Criterion for selecting feature at each node)
Max Depth - [None, 2, 3, 5, 10] Depth of the trees built
min_samples_split - [None,5,10] minimum samples at a node to split it
min_samples_leaf - [None,3,5] minimum samples required in a leaf

We scale the data using StandardScaler and perform a 80-20 split into training and "test" data. This out of the bagGridSearch CV works reasonably fast on our dataset (~ 4 mins). The accuracy of this out of the bag random forest classifier gives us a "test" accuracy of

Accuracy : 0.959

We see a similar accuracy to the predictions on the real "blind" test data on Kaggle. Since RF's work reasonably quickly we use them to explore other ideas

### One vs All Classification
Let's take a look at the confusion matrix which we show below
![Alt text](/Figures/Confusion_Matrix.png?raw=true "confusion matrix")

As we can see 7 gets often misclassified as 9 and 3 gets misclassified often as well. As an example the classification accuracy of 3 is 0.93.  However would One vs All Classification perform better? 

If we perform One vs All Classification on the number 3 then we get accuracy of 0.98. At first glance One vs All Classification works a lot better. However since on the real test set we don't know what the digit is, we have to train 10 separate classifier for one vs all classification for each digit. Then we can predict the probability of the number belonging to a given class by running the test data through all the 10 classifiers and find the class with the max probability. Performing the One vs All classification in the "test" data we get

Accuracy = 0.96

The gains in accuracy are negligible and not enough to justify a One vs All classification. Thus we abandon this path

## Dimensionality reduction with SVD
Since each image has 784pixels, the dimensionality of our feature space is quite large. We can perform a singular value decomposition of our data to reduce the dimensionality to make our models train faster. Also working in a lower dimensional space can reduce "noise". We perform the SVD on our dataset. The cumsum of the variance captured vs number of features is shown below. The first 400 features capture more than 90% of the variance

![Alt text](/Figures/svd_info.png?raw=true "Signal")

Below is the reconstruction of the digits using the first 10,100, 200 and 784 principle components

![Alt text](/Figures/svd_recon.png?raw=true "Signal")


In the interest of faster computational time we keep only the first 200 components. Fitting the same GridSearchCV RF model described above to the transformed dataset projected onto the 200 principle components, we get

Accuracy = 0.91

This is lower than what we got using all the pixels in our Out of Bag RF model. Interestingly RF also takes longer to run on the SVD generated dimensionally reduced data. This is contrary to what is expected and the reason is still being explored.

## SVM
We use Support Vector Machines (SVMs) to classify our data. The SVM used to fit the training data is also optimized using a GridSearchCV. The parameters for the grid search are given below. We use the "rbf" kernel

C_range = [0.01,0.1,1,10,100]
gamma_range =  ['auto',  'scale']

SVM is a margin classifier and tries to fit a classification boundary (hyperplane) by transforming low dimensional data into high dimensional data using the kernel trick and calculating the pairwise distance between individual datapoints. Thus data with large number of datapoints can slow down SVM drastically. We performed a similat 80-20 split on our training data. Doing the full SVM takes about 5 hours with 5-fold cross-validation, considerably longer than taken by Random Forest algorithm. We then get

Accuracy = 0.965

Although we get slightly better accuracy than Random Forests it comes at the expense of increased computational time.

To remedy this we can use the first 200 principle components of the transformed dataset we generated using the SVD. The computational time is then brought down to about half an hour with 2-fold cross-validation. Surprisingly this gives us a better accuracy of

Accuracy = 0.972

However the computational time of about half an hour is still too large. The reason for this is the > 30,000 data points we used for training. Most of the digits are well classified and do not lie near the SVM classification boundary and have no impact on the construction of the classification boundary, We can improve upon this only using the data which we suspect is near the classification boundary. This will be the data whose prediction probabilities will be comparable between two classes.

We consider the max probability of a given training data point of belonging to a certain class. For all training examples whose maximum probability of belonging to a class is < 0.8, we train a SVM model on them. This speeds up the SVM since it's being performed on a smaller set of data. We then use the SVM model to fit only the test points who also have a maximum probability of < 0.8 of belonging to a given class.  Then combining with the samples that we well classified by Random Forests, we get

Accuracy = 0.965

This is better than accuracy just due to Random Forests. Howver it is still lower than the dimensionally reduced SVM

## XGBoost
We also try to use extreme Gradient Boosting for our digit classification problem.  XGB is extremely slow for fitting over the entire dataset with all 784 pixels as features.  Thus we take the dimensionally reduced dataset that we obtained by performing the SVD and use the first 200 features only. We use GridSearchCV to determine the best fit parameters with parameters given below

max_depth: range (2, 10, 2),
n_estimators: range(20, 120, 20),
learning_rate: [0.001,0.003,0.01, 0.03, 0.1]

We do a 80-20 split on the training data and use 2-fold cross-validation. XGBoost takes an hour to run and gives us 

Accuracy = 0.95

which is still lower than SVM while taking longer.

## Conclusions
We achieved a maximum score of 0.97 in our classification. This is considerably poorer than CNN's which reliably perform better than 0.98 accuracy with more or less out of the box implementation. For possible future improvements, a voting classifier with RF, XGB and SVM voting together tried as well as stacking classifier should be implemented.


## Blog

https://aghalsasi-datascience.blogspot.com/2021/03/mnist-digit-recognition.html

## Navigating the repository

The main analysis and presentation notebook is Prediction Digits. Some analyses take quite some time to run.