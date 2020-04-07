r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""

**Increasing k improves the generalization for unseen data up to a point (k=5 in our case),
and beyond that the accuracy decreases as k grows.
We assume that increasing k results in reducing the noise effect on the predictions,
but increasing it to much (in proportion to the size of the dataset) you might not get
close enough examples to the instance to create a good prediction.**

"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""

the selection of delta is arbitrary, because we did not add the regularization term.
so increasing delta, and multiplying the the weight matrix by the same factor will give us the same results,
as the gradient has no direct dependencies on W, only on the samples, and the sign of the hinge function(which 
depends on W, but does not affect the magnitude).

if we would have introduced a regularization term, we would have been penalized by large values of **W**, and the value 
of $\delta$ would have to be normalized, but as we did not add it, we are free to "stretch" the differences between
the produced values, and pass the $\delta$ barrier. (by multiplying the weights with some constant, $\alpha$)
 
"""

part3_q2 = r"""

1. From the images of the weight's matrix, for each of the classes(different digits)
it is clear that the model Identifies the areas in the picture that are bright in the train samples,
and gives the most weight to the ones that are most common in the training set. The higher the weights in a given area
the more significant the pixels in the area. Then, when it is given unseen data, it find the weight matrix that fits
the best the bright pixels of the data, and makes a prediction based on that.

we can see that the pictures that are labeled incorrectly are digits that are written disproportionally, or rotated,
and thus the pixels that are bright are not in the "regular" place, and the models prediction depends on the absolute 
location of the bright pixels, so it predicts the wrong results  

2. it is similar to knn, as both methods use similarities to the training data, to predict the class of sample.
but, unlike knn, this model produces a function that predicts the class, instead of just finding the closest neighbors.
Also not every feature is as significant, some are more and some are less, unlike knn, where every feature is taken evenly
when the distance is calculated(weighted "distance")

"""

part3_q3 = r"""

1. Good, but I might have changed it a bit during the epochs. It the accuracy keeps improving throughout the epochs,
so it is not too big, as if it was too big in the end we would see some ups and downs, from "passing" the minimum point.
it is not to small, because we did make progress, and reached a plateau pretty quickly, and it seems that we are 
converging to the optimal weights, with some negligible difference. if it was too small it would not have plateaued that
way.
 

2. Slightly overfitted to the training set - because the training set accuracy is larger then the testing set accuracy,
and we can see that the accuracy in the training set increases more then the accuracy in the test set.
it is not highly overfitted, because the accuracy in the test set does not get worse, and it is not under fitted,
because the accuracy in the training set is >90, and because the data is not perfectly linearly separable, we expect 
some misses.
 
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
The Ideal residual plot will be such that the points mean distance from the x-axis will be zero, which means that
the expectancy of our prediction is the same as of the test set values. The more symmetric it is around the x-axis, and 
close to it means that our predictions are better.

It seems that the two plots are very similar, except for data points with low y values. this is where the last model
preforms better then the purely linear model, also the points are more tightly packed near the x-axis with the last
model, so it seems better fitted
"""

part4_q2 = r"""

1. The choice of logspace instead of linespace is better here, as the purpose of $\lambda$ in the loss function is to
regulate for too big values of $||w||^2$, which will imply overfitting of the model(as it means that the function is too
complex). In order for lambda to have the correct effect on the result it needs to be in a certain order of magnitude.
too big and it will dominate the term, too small and it will have too small of an effect. this is why it is important 
for it to be in the same order of magnitude as of the optimal $||w||^2$, but when we take line space we dont move too 
much with respect to orders of magnitude, and stay in the same one (as long as there are not many elements in the list)

2. we preformed grid search with the hyper-parameters, which means that we tried every possible combination of both,
that is why we took $n_\lambda$ times $n_{degree}$ sets of hyper parameters = 60.
for every set we used 3-folds, which means 3 fitting and testing for each set of hyper-parameters, is 60*3=180 

"""

# ==============
