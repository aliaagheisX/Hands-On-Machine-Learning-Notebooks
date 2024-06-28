# Classification
Now we will turn our attention to classification systems.

## 01 Notebook (Binary Classification)

Here my `Notes` & `Exploration` to:
- ``MINST`` Dataset
- More on Cross Validation
- More metrics (Precision, Recall, ROC Curve)
- How to Tune thresholds to increase one metric (eg., Precision)
- How to compare models using `ROC AUC`
- Some notes on how to handle inbalanced dataset

## 02 Notebook (Multiclass Classifier)
Here my `Notes` & `Exploration` to new type of classifier:

- Know more about `OvO`(one versus one) or `OvR`(one versus rest) 
- How to analyze Error that made by model
- Some notes on `SVM` (support vector machine)
    - it scale poorly with size of dataset (that's why `OvO` work well with it)

## 03 Notebook (Other Classifiers)
Here my `Notes` & `Exploration` to new type of classifiers:

### 1. Multilabe Classifier
> such classification that output multiple binary classifications
- eg., face recogniation of 3 faces, input will be image and output will be (Alice, Bob, Not Jon) [1, 0, 1]

- I learned how to analyze error & how to measure performace
```python
f1_score(y_train_multilabel, y_train_knn_pred, average="macro")
f1_score(y_train_multilabel, y_train_knn_pred, average="weighted")
```

    - `macro` will return average/mean of F1_score of each label
    - `weighted` will return weighted mean of F1_score of each label
    - if there's class repeated more will give more importance
    - eg., if there's many faces of "Alice" 

### 2. Multioutput Classifier
its more genreal case of `multilabel classification` where each label can have multiple class
> eg.,
> noise removal system where foreach label output intensity\
> multilabel as there's label for each pixel\
> multioutput as there's different output for each label \
> (here there's thin line between regression & classification) as predict intensity will be more akin to regression


## 03 Notebook (Exercise)
contain my solutions to first two excerices.
### Ex1
<div style="background-color:#3871b2;color:#FFF;font-size:18px;padding:20px;width:60%;border-radius:20px"> Try to build a classifier for the MNIST dataset that achieves over
97% accuracy on the test set. Hint: the KNeighborsClassifier
works quite well for this task; you just need to find good
hyperparameter values (try a grid search on the weights and
n_neighbors hyperparameters).</div>


### Ex2
<div style="background-color:#3871b2;color:#FFF;font-size:18px;padding:20px;width:60%;border-radius:20px">
    Write a function that can shift an MNIST image in any direction
    
(left, right, up, or down) by one pixel. Then, for each image in    
the training set, create four shifted copies (one per direction) and
add them to the training set. Finally, train your best model on this
expanded training set and measure its accuracy on the test set.
You should observe that your model performs even better now!
This technique of artificially growing the training set is called
data augmentation or training set expansion.
</div>

## 04-05 Notebook (🚢 Tackle the Titanic Dataset)
- The first notebook is EDA notebook to titanic competition  
- The second one contain Pipeline, Model Exploration, Model Submission