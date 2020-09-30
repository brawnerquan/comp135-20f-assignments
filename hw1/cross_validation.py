import numpy as np


import os
import warnings

import sklearn.preprocessing
import sklearn.pipeline
import sklearn.linear_model
import sklearn.neighbors
import sklearn.model_selection

from performance_metrics import calc_mean_squared_error


def train_models_and_calc_scores_for_n_fold_cv(
        estimator, x_NF, y_N, n_folds=3, random_state=0):
    ''' Perform n-fold cross validation for a specific sklearn estimator object

    Args
    ----
    estimator : any regressor object with sklearn-like API
        Supports 'fit' and 'predict' methods.
    x_NF : 2D numpy array, shape (n_examples, n_features) = (N, F)
        Input measurements ("features") for all examples of interest.
        Each row is a feature vector for one example.
    y_N : 1D numpy array, shape (n_examples,)
        Output measurements ("responses") for all examples of interest.
        Each row is a scalar response for one example.
    n_folds : int
        Number of folds to divide provided dataset into.
    random_state : int or numpy.RandomState instance
        Allows reproducible random splits.

    Returns
    -------
    train_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for train set for fold f
    test_error_per_fold : 1D numpy array, size n_folds
        One entry per fold
        Entry f gives the error computed for test set for fold f

    Examples
    --------
    # Create simple dataset of N examples where y given x
    # is perfectly explained by a linear regression model
    >>> N = 101
    >>> n_folds = 7
    >>> x_N3 = np.random.RandomState(0).rand(N, 3)
    >>> y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
    >>> y_N.shape
    (101,)

    >>> import sklearn.linear_model
    >>> my_regr = sklearn.linear_model.LinearRegression()
    >>> tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(
    ...                 my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)

    # Training error should be indistiguishable from zero
    >>> np.array2string(tr_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'

    # Testing error should be indistinguishable from zero
    >>> np.array2string(te_K, precision=8, suppress_small=True)
    '[0. 0. 0. 0. 0. 0. 0.]'
    '''

    assert (n_folds < 2)

    train_error_per_fold = np.zeros(n_folds, dtype=np.float64)
    test_error_per_fold = np.zeros(n_folds, dtype=np.float64)

    # TODO define the folds here by calling your function
    # e.g. ... = make_train_and_test_row_ids_for_n_fold_cv(...)
    train, test = (make_train_and_test_row_ids_for_n_fold_cv(x_NF.shape[0], n_folds, random_state))
    #get training set from fold
    # print(train)
    # print(x_NF.shape)
    # print(test)
    for i in range(n_folds):
        FL = len(train[i])
        F = x_NF.shape[1]
        TL = len(test[i])
        current_fold = np.empty([FL, F]);
        current_fold_y = np.empty([FL]);
        to_predict = np.empty([TL,F]);
        to_predict_y = np.empty([TL]);
        for j in range(FL):
            current_fold[j] = x_NF[train[i][j]]
            current_fold_y[j] = y_N[train[i][j]]
        # print("FOLD: ", current_fold)
        for k in range(TL):
            to_predict[k] = x_NF[test[i][k]]
            to_predict_y[k] = y_N[test[i][k]]
        estimator.fit(current_fold, current_fold_y)
        yhat_train = estimator.predict(current_fold)
        # print(yhat_train)
        yhat_test = estimator.predict(to_predict)
        # print(yhat_test)
        train_error_per_fold[i] = calc_mean_squared_error(current_fold_y, yhat_train)
        test_error_per_fold[i] = calc_mean_squared_error(to_predict_y, yhat_test)


    #fit each fold, predict each fold, then calculate error
    #call predict on train and test and calculate error append error to
    #respective list
    # TODO loop over folds and compute the train and test error
    # for the provided estimator

    return train_error_per_fold, test_error_per_fold


def make_train_and_test_row_ids_for_n_fold_cv(
        n_examples=0, n_folds=3, random_state=0):
    ''' Divide row ids into train and test sets for n-fold cross validation.

    Will *shuffle* the row ids via a pseudorandom number generator before
    dividing into folds.

    Args
    ----
    n_examples : int
        Total number of examples to allocate into train/test sets
    n_folds : int
        Number of folds requested
    random_state : int or numpy RandomState object
        Pseudorandom number generator (or seed) for reproducibility

    Returns
    -------
    train_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N
    test_ids_per_fold : list of 1D np.arrays
        One entry per fold
        Each entry is a 1-dim numpy array of unique integers between 0 to N

    Guarantees for Return Values
    ----------------------------
    Across all folds, guarantee that no two folds put same object in test set.
    For each fold f, we need to guarantee:
    * The *union* of train_ids_per_fold[f] and test_ids_per_fold[f]
    is equal to [0, 1, ... N-1]
    * The *intersection* of the two is the empty set
    * The total size of train and test ids for any fold is equal to N

    Examples
    --------
    >>> N = 11
    >>> n_folds = 3
    >>> tr_ids_per_fold, te_ids_per_fold = (
    ...     make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
    >>> len(tr_ids_per_fold)
    3

    # Count of items in training sets
    >>> np.sort([len(tr) for tr in tr_ids_per_fold])
    array([7, 7, 8])

    # Count of items in the test sets
    >>> np.sort([len(te) for te in te_ids_per_fold])
    array([3, 4, 4])

    # Test ids should uniquely cover the interval [0, N)
    >>> np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])

    # Train ids should cover the interval [0, N) TWICE
    >>> np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)]))
    array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
            8,  9,  9, 10, 10])
    '''
    assert (n_folds < 2)

    if hasattr(random_state, 'rand'):
        # Handle case where provided random_state is a random generator
        # (e.g. has methods rand() and randn())
        random_state = random_state # just remind us we use the passed-in value
    else:
        # Handle case where we pass "seed" for a PRNG as an integer
        random_state = np.random.RandomState(int(random_state))

    # TODO obtain a shuffled order of the n_examples
    row_ids = np.asarray(list(range(n_examples)))
    random_state.shuffle(row_ids)
    test_ids_per_fold = list(np.array_split(row_ids, n_folds))
    # print(test_ids_per_fold)
    train_ids_per_fold = list();
    for i in range(len(test_ids_per_fold)):
        train_ids_per_fold.append(np.array(list(set(row_ids) ^ set(test_ids_per_fold[i]))))
    # print(train_ids_per_fold)
    #hstack
    # TODO establish the row ids that belong to each fold's
    # train subset and test subse


    return train_ids_per_fold, test_ids_per_fold

def make_poly_linear_regr_pipeline(degree=1):
    pipeline = sklearn.pipeline.Pipeline(
        steps=[
         ('rescaler', sklearn.preprocessing.MinMaxScaler()),
         ('poly_transformer', sklearn.preprocessing.PolynomialFeatures(degree=degree, include_bias=False)),
         ('linear_regr', sklearn.linear_model.LinearRegression()),
        ])

    # Return the constructed pipeline
    # We can treat it as if it has a 'regression' API
    # e.g. a fit and a predict method
    return pipeline


# N = 47
# n_folds = 9
# tr_ids_per_fold, te_ids_per_fold = (make_train_and_test_row_ids_for_n_fold_cv(N, n_folds))
# len(tr_ids_per_fold)
# # 3
# print("TRAINING IDS: ", tr_ids_per_fold)
# print("TEST IDS: ", te_ids_per_fold)
# # Count of items in training sets
# print(np.sort([len(tr) for tr in tr_ids_per_fold]))
# # array([7, 7, 8])
#
# # Count of items in the test sets
# print(np.sort([len(te) for te in te_ids_per_fold]))
# # array([3, 4, 4])
#
# # Test ids should uniquely cover the interval [0, N)
# print(np.sort(np.hstack([te_ids_per_fold[f] for f in range(n_folds)])))
# #array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10])
#
# # Train ids should cover the interval [0, N) TWICE
# print(np.sort(np.hstack([tr_ids_per_fold[f] for f in range(n_folds)])))
# # array([ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5,  6,  6,  7,  7,  8,
#     # 8,  9,  9, 10, 10])



# Create simple dataset of N examples where y given x
# is perfectly explained by a linear regression model
# N = 10
# n_folds = 7
# x_N3 = np.random.RandomState(0).rand(N, 10)
# y_N = np.dot(x_N3, np.asarray([1., -2.0, 3.0])) - 1.3337
# print(y_N.shape)
# # (101,)
#
# import sklearn.linear_model
# my_regr = sklearn.linear_model.LinearRegression()
# tr_K, te_K = train_models_and_calc_scores_for_n_fold_cv(my_regr, x_N3, y_N, n_folds=n_folds, random_state=0)
#
# # Training error should be indistiguishable from zero
#
# print(np.array2string(tr_K, precision=8, suppress_small=True))
# # '[0. 0. 0. 0. 0. 0. 0.]'
# # Testing error should be indistinguishable from zero
# print(np.array2string(te_K, precision=8, suppress_small=True))
# # '[0. 0. 0. 0. 0. 0. 0.]'
