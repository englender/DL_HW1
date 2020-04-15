r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer: 
Increasing the k up to a certain value will improve generalization. This happens because the increment of 
the k value helps avoid giving a lot of weight to noise samples in the classification process. If we increase the k 
value too much, we will give weight to samples that are far from the test sample, and might make our prediction wrong.
In our case the best k value is 3, which gave us accuracy of 92%.

"""

part2_q2 = r"""
**Your answer: **
1. When training on the entire train-set and selecting a model with respect to the train-set accuracy, we risk 
overfitting our model. Because we are trying to choose the model's hyperparameters based on the same data that we train 
the model on, we will ensure that the model will work perfectlyon the train-set but this will reduce the model 
generalization for unseen data. In conclusion, using k-fold CV is better than this method.

2. When selecting the best model with respect to the test-set accuracy, we ensure that our model will work the best on 
our specific given test-set, and once again we risk in overfitting our model to a specific test-set and we will reduce 
the model generalization for unseen data. In other words, when we want to check if our model has good performance we 
check the test-set and get high results because we calibrated the model based on the same set, but we can
anticipate that we will achieve much lower results when facing unseen data.


"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**
The hyperparameters delta and gamma both have influence on the loss function. When enlarging/shrinking the delta 
parameter,we enlarge/shrink the weight and the scores differences, meaning we control the first argument of the SVM loss 
function (the margin loss). On the other hand, when changing the gamma parameter we control the second argument of the 
SVM loss function (the regularization loss). Both of this parameters control L(W) so we can set the delta arbitrary
because for every chosen delta value we can control the L(W) to meet our needs through the gamma param.
 
"""

part3_q2 = r"""
**Your answer:**
1. Given the images from the visualization section we can see that through the training process of the linear model, for 
each class we give a higher weight to areas of the pictures that are commonly brighter, based on the relevant train 
samples of that class. So when the classifier is given unseen data, it will classify the sample using the weight matrix,
and depending on the bright areas of the new sample it will classify to the class that best fits it.
Some of the classification errors can be explained by that the hand writing of wrongly classified samples is messy or 
disproportioned and as a result the brighter areas are not in the expected places thus resulting in a wrong 
classification. However, we can notice that most of the errors are between classes that similar to begin with, for 
example: the digits 5 and 6 have a similar bright areas distribution (mostly bright on the bottom and dark on the top),
so their weights are supposed to be similar.

2. The similarity of this interpretation and the knn model is that both models classify based on the most similar 
training samples. The first difference of the two implementations is that knn model compare the test sample to all 
training samples in the classification process (calculating the dist) and the linear classifier does it using the 
weight matrix alone. The second difference is that knn model takes in account every feature evenly as appose the linear
classifier that gives a different weight to every feature.

"""

part3_q3 = r"""
**Your answer:**
1. Based on the graph of the training set loss, we would say that the learning rate we chose is good. If we were to 
choose a higher learning rate then the loss graph would probably have spikes in its decreasing rate, meaning that we 
would expect to see some momentary increases of the loss value. This is because we are making too large step towards the 
gradient direction, and passing the minimum value. 
If we were to choose a lower learning rate the descent rate of the loss function will decrease and as a result the final
loss would be not as good as with a higher learning rate.

2. Based on the accuracy graph we would say that our model is slightly overfitted to the training set. We can see that 
we get a slightly better accuracy for the training set then the valid set, meaning that the model generalization is
slightly limited because of this overfit.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
The ideal pattern to see in a residual plot is a plot that all points are near the x axis, meaning that for a single 
sample the difference between the prediction of y value to its ground truth is as small as possible.

We can see based on the residual plot that we got that the fitness of the last model is better than fitness of the 
top 5 features model. The data points of the last model plot are compressed more tightly around the x axis than the 
previous one. Also we can see that the maximum difference between the prediction and the ground truth is bigger on the 
top 5 features model and that there are more data points with a larger difference.

"""

part4_q2 = r"""
**Your answer:**
1. When using logspace instead of linspace we scan the entire lambda range, but most of our focus is on smaller values.
The lambda hyperparameter controls the importance of the regularization factor - the smaller this value is the risk for
complex loss function is greater, thus the risk for overfitting is greater. Therefore, if we get the best results by CV
for the loss function with smaller lambda value, we can conclude that the model generalization for unseen data will 
increase. However, if we didn't get good results for smaller value, then we will try the bigger values, and that why
the use og logspace will fit in this case.

2. For each sets of parameters we fit our model for every fold, therefor for each set of parameters we fit the model
k_folds times. The parameters we checked in our CV are lambda and degree, so the number of parameters set is the number 
of degrees possible times the number of lambda possible, meaning degree_range*lambda_range. In total we fit our model
k_folds * degree_range * lambda_range, and in our case its 3*3*20=180.

"""

# ==============
