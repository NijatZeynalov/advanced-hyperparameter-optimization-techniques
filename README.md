## 1. Basic techniques (random search, grid search, halving)

Hyperparameter optimization in machine learning intends to find the hyperparameters of a given machine learning algorithm that deliver the best performance as measured on a validation set.
While model parameters are learned during training — such as the slope and intercept in a linear regression — hyperparameters must be set by the data scientist before training.

The core algorithms for hyperparameter optimization, found in the Scikit-learn package, are grid search and random search. Recently, the Scikit-learn contributors have also added the halving algorithm to improve the performances of both grid search and random search strategies.


### 1.1 Grid search

Grid search is a tuning technique that attempts to compute the optimum values of hyperparameters. It is an exhaustive search that is performed on a the specific parameter values of a model. The model is also known as an estimator.

Basically, we divide the domain of the hyperparameters into a discrete grid. Then, we try every combination of values of this grid, calculating some performance metrics using cross-validation. The point of the grid that maximizes the average value in cross-validation, is the optimal combination of values for the hyperparameters.



![alt text](https://www.yourdatateacher.com/wp-content/uploads/2021/03/image-6.png)


### 1.2 Random Search

Random search is similar to grid search, but instead of using all the points in the grid, it tests only a randomly selected subset of these points. The smaller this subset, the faster but less accurate the optimization. The larger this dataset, the more accurate the optimization but the closer to a grid search.

Random search is a very useful option when you have several hyperparameters with a fine-grained grid of values. Using a subset made by 5-100 randomly selected points, we are able to get a reasonably good set of values of the hyperparameters. It will not likely be the best point, but it can still be a good set of values that gives us a good model.


![alt text](https://www.yourdatateacher.com/wp-content/uploads/2021/03/image-7.png)

## 2. Successive halving search.

Grid search and random search work in an uninformed way: if some tests find out that certain hyperparameters do not impact the result or that certain value intervals are ineffective, the information is not propagated to the following searches.

Scikit-learn has recently introduced the HalvingGridSearchCV and HalvingRandomSearchCV estimators, which can be
used to search a parameter space using successive halving applied to the grid search and random search tuning strategies.


## 3. Bayesian optimization using scikit-optimize.

## 4. Customizing Bayesian optimization.

## 5. KerasTuner: hyperparameter tuning for Keras models (Bayesian optimization).

## 6. KerasTuner: hyperparameter tuning for Keras models (hyperband optimization).

## 7. Parameter cheetsheet for hyperparameter optimization

## 8. Resources and references


## Project Structure


> Folder structure options and naming conventions for software projects

### A typical top-level directory layout

    .
    ├── data                   # Compiled files (alternatively `dist`)
    ├── docs                    # Documentation files (alternatively `doc`)
    ├── basic_techniques.ipynb                     # Source files (alternatively `lib` or `app`)
    ├── bayesian_optimization.ipynb                    # Automated tests (alternatively `spec` or `tests`)
    ├── keras_tuner.ipynb                   # Tools and utilities
    ├── LICENSE
    └── README.md

> Use short lowercase names at least for the top-level files and folders except
> `LICENSE`, `README.md`
