# Hands on ML

# Chapter 1

### Types of ML

- Supervised vs Unsupervised vs semi-supervised vs Reinforcement Learning
    - unsupervised (Clustering - Visualization and dimensionality reduction - Association rule learning)
    - semi ⇒ deep belief network (DBN)
    - Reinforcement ⇒ agent (learning system) get rewards or penalty
        - many robots implement Reinforcement Learning algorithms to learn how to walk.
- Online vs Batch
    - online ⇒ can added data later to enhance training (*learning incrementally*)
        - by mini batches
        - used to train dataset that cannot fit on one machine (*Out-of-core learning*)
        - how fast should adapt to changing data ⇒ learning rate ⇒ BUT = how fast will forget old data
            - too large ⇒ very sensitive to noise or new data
            - too small ⇒ large inertia ⇒ too  slow
        - if system keep fed by bad data ⇒ system will keep gradually decrease
    - batch ⇒ to train on new data must retrain from zero
- Instance based vs Model based
    - instance ⇒ memorize examples by heart and then return on similarity
    - model ⇒ create model

### Main Challenges

### Data Errors

- Insufficient Quantity of Training Data
    - on study ⇒ the basic models can reach on adv model if have sufficient data
    - so we can throw all money on collecting data
    - but get more data is still very costly than advance algo
- Non representative Training Data
    - too small data ⇒ may result ⇒ noisy data (***sampling noise***)
    - very large samples ⇒ may result ⇒ sampling method is flawed (***sampling bias***)
    - when some of sampling not response ⇒ (***non response bias***)
- Poor Quality Data
    - if data is full of errors, outliers and noise ⇒ harder to detect patterns
    - its worthy to spend time on cleaning data
        - like remove obvious outliers manually
        - missing few feature ⇒ remove feature || ignore sample
         || fill it || train with feature and one without
- Irrelevant Features
    - model = good ⇒ if training data contains
        - enough relevant features
        - & not too many irrelevant ones
    - **Feature engineering**
        - *Feature selection*
        - *Feature extraction* ⇒ combining existing features to produce useful one ⇒ dimensionality reduction
        - create new feature by gather new data

### Algorithm Errors

- Overfitting the training Data
    - perform well on train data but it does not generalize well
    - happen when model too complex relative to amount and noisiness of data
    - solutions
        - simplify model ⇒
            - reducing parameters
            - reducing features enter model
            - constraining model (***regularization***)
        - collect more training data
        - reduce noise in training data
- Underfitting the training Data
    - too simple even on training data
    - solutions
        - select more powerful model with more parameter
        - feed better feature
        - decrease regularization hyperparameter (reduce constrain on model)

### Testing & Validation

- Testing
    - instead of hoping that the model will generalize well on new data ⇒ split to test set
        - error rate in new cases (generalization error)(out of sample error)
        - if train error is low **BUT** generalization error is high ⇒ overfit
- Validation (holdout/cross)
    - holdout validation ⇒ at last train step ⇒ best model on both (train & val)
        - too small val ⇒ may not select best model
        - too large val ⇒ train set @validation step become small ⇒ like select best bike for car race
    - cross validation ⇒ each model evaluate once at each val set ⇒ training time increase
- Data Mismatch
    - data of val, test sets must be exclusive representative on production
        - so you may collect them by hand
        - and then shuffle and ensure no duplicates between two sets
    - if you collect train set from different sources
        - if behave bad on test/val set 
        ⇒ you can’t know for sure if its (overfit || mismatch)
        - sol ⇒ holdout some of train set ⇒ called ***train-dev set***
        if behave well on it ⇒ data mismatch ⇒ try to make train similar
        if behave bad on it ⇒ overfit ⇒ regularization, get more train data
- NFL (no free lunch theorem) ⇒ you must make reasonable assumptions

# Chapter 2

## 1. Look at Big Picture

1. what’s objective of model ? to decide ⇒   
    - which algorithm used
    - which performance measure you will use to evaluate your model
    - how much effort you will **spend tweaking** it
2. what the current solution looks like (if any) ?
    - will give you reference of performance
3. Frame the problem (supervised / unsupervised, online/batch, etc)
4. Select performance measure
    - **RMSE** (root mean square error) (Euclidean norm) (l_2 norm)
    - **MAE** (average absolute deviation) (Manhattan norm) (l_1 norm)
        - good “when there’s many outlier”
    - More generally: l_k of vector v = (|v_0|^k + |v_1|^k + |v_2|^k +…)^(1/k).l_0
        - the higher **k** ⇒ the more focus on large values and ignore small values
        - that’s why l_2 more sensitive to noise
5. Check the Assumptions

## 2. Get Data

### 1. Look at data (don’t look to much 🫣snoop bias)

```python

```

- some features are capped (it’s okay)
- the predicated attribute is capped @500,000 ⇒ danger ⇒ ML model may predict it never get larger than that
    - you can remove this instances
    - or collect proper data

### 2. Create Test set

- pure randomize
    
    ⇒ good with large data sets
    
    - implementation
        - you may randomize  ⇒ but ⇒ every run will select different dataset
        - you may randomize one and save it ⇒ but ⇒ what about when editing data
        - you can hash  and select ratio on <test_ration ⇒ ✅
            - just use scikit_learn
            
            ```python
            sklearn.model_selection import train_test_split 
            ```
            
- ⇒ risk of introducing a significant **sampling bias** in not very large dataset
    - dataset must be representative of population
    - if  population 51% female & 49% male ⇒ the dataset should be same
    
    ⇒ this called **stratified sampling**
    
    - population divide into ⇒ homogeneous groups called **strata**
    - and right amount sampled from each **strata**
- there’s other options for sampling dataset (eg: cluster sampling)

### Check Notebook ….

### Scikit API Design

- **Consistency of objects**
    
    all have same simple interface
    
    - ***Estimators***
        - object can estimate some parameter | based on dataset
        - the estimation itself ⇒ `fit()`
            - it only need datasets (one or two in case supervised)
            - others is hyperparameters & called in constructor
        - eg: `imputer`
    - ***Transformers***
        
        some estimator ⇒ can also ⇒ **transform dataset**    
        
        - have `fit_transform()` which is equivalent to `fit()` but sometimes be more optimized
        - also have `tranform(_dataset)` that return the new dataset
    - ***Predictors***
        
        some estimator ⇒ given dataset can ⇒ predict ****
        
        - have `predict()` **take** dataset of new instances & **return**  dataset of new prediction
        - also `score()` measure quality of prediction **given** test set
        
- **Inspection**
    - all hyperparameters can be accessed as direct attributes eg: `imputer.strategy`
    - all learned parameters can be accessed directly with suffix at end_ eg: `imputer.statistics_`
- **Nonproliferation of classes ⇒** all datasets are either (NumPy | SciPy sparse) matrix
- **Composition ⇒** existing blocks are used as much as possible
- **Sensible defaults ⇒** provides reasonable default values for most parameters