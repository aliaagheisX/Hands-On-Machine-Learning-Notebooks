# Exercises

1. Which Linear Regression training algorithm can you use if you have
a training set with millions of features?

> Mini-Batch Gradient Descent or Schostic, can't use Batch GD or Closed Form

> ___Correction:___ you can use Batch if it's fit memory __But__ can't use SVD, Normal form as it growth quadratic
 
2. Suppose the features in your training set have very different scales.
Which algorithms might suffer from this, and how? What can you
do about it?

> all gradient descent algorithms, they will get slower to converge

> to solve it use `standardScalar` or `minMaxScalar`

3. Can Gradient Descent get stuck in a local minimum when training a
Logistic Regression model?

> No as Softmax, Sigmoid both are convex functions

4. Do all Gradient Descent algorithms lead to the same model,
provided you let them run long enough?

> Kinda as you may need to gradually decrease $\eta$ (learning-rate) for Stochasitc, Mini-Batch to convege

5. Suppose you use Batch Gradient Descent and you plot the validation
error at every epoch. If you notice that the validation error
consistently goes up, what is likely going on? How can you fix this?

> The Model is to complex that already _overfit_ training data (have high Variance)

> to solve it, you need to make model simpler (regularize it)

6. Is it a good idea to stop Mini-batch Gradient Descent immediately
when the validation error goes up?

> No, as it the curve is bumbier and not smooth, so need to take time to ensure its really ended

7. Which Gradient Descent algorithm (among those we discussed) will
reach the vicinity of the optimal solution the fastest? Which will
actually converge? How can you make the others converge as well?

> Stochastic -> will probably get to global minimum faster than other

> Batch GD -> will converge for sure

> for others to converge, we could gradullay decrease $\eta$ learning rate

8. Suppose you are using Polynomial Regression. You plot the learning curves and you notice that there is a large gap between the training
error and the validation error. What is happening? What are three ways to solve this?

> The model has high variance (sensitive to little variation in data) (complex model)

> Solution is to regularization (make model simpler) 
> 1. `Ridge Regression` add l2 penalty to weights
> 2. `Lasso Regression` add l1 penalty to weights
> 3. `Elastic Net` and l1, l2 penalty to weights with some ration
> 4. `Early Stopping` stop when validation error start to go up

9. Suppose you are using Ridge Regression and you notice that the
training error and the validation error are almost equal and fairly
high. Would you say that the model suffers from high bias or high
variance? Should you increase the regularization hyperparameter α
or reduce it?

> it's has `high bias` for sure (model bias to assumption that's wron) (model is too simple)

> will decrease regularization parameter $\alpha$


10. Why would you want to use:

    a. Ridge Regression instead of plain Linear Regression (i.e., without any regularization)?
    > always, it's nice to have a little bit of regularization
    
    b. Lasso instead of Ridge Regression?
    > when you suspect that there's few useful Features and you want your model to automatically remove useless features 

    c. Elastic Net instead of Lasso?
    > most of time, Lasso behave randomly when #features > #samples, and when exist many strongly correlated features


11. Suppose you want to classify pictures as outdoor/indoor and
daytime/nighttime. Should you implement two Logistic Regression
classifiers or one Softmax Regression classifier?
> `Two Logistic Regression` as the __softmax__ is multi-nomial (can classify more than 1 class) not _multi-output_ (can output multiple classes)  

12. Implement Batch Gradient Descent with early stopping for Softmax
Regression (without using Scikit-Learn).
> already done in notebook