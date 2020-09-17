'''
hw0.py
Author: TODO

Tufts COMP 135 Intro ML, Fall 2020

Summary
-------
Complete the problems below to demonstrate your mastery of numerical Python.
Submit to the autograder to be evaluated.
You can do a basic check of the doctests via:
$ python -m doctest hw0.py
'''

import numpy as np


def split_into_train_and_test(x_all_LF, frac_test=0.5, random_state=None):
    ''' Divide provided array into train and test sets along first dimension

    User can provide random number generator object to ensure reproducibility.

    Args
    ----
    x_all_LF : 2D np.array, shape = (n_total_examples, n_features) (L, F)
        Each row is a feature vector
    frac_test : float, fraction between 0 and 1
        Indicates fraction of all L examples to allocate to the "test" set
        Returned test set will round UP if frac_test * L is not an integer.
        e.g. if L = 10 and frac_test = 0.31, then test set has N=4 examples
    random_state : np.random.RandomState instance or integer or None
        If int, will create RandomState instance with provided value as seed
        If None, defaults to current numpy random number generator np.random.
    '''

    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    x_test_NF = x_all_LF.copy()
    random_state.shuffle(x_test_NF)


    test_dimensions = int(np.ceil(x_all_LF.shape[0] * frac_test))

    x_test_NF_new = x_test_NF[x_test_NF.shape[0] - test_dimensions: :, :]
    x_train_MF = x_test_NF[0: x_test_NF.shape[0] - test_dimensions, :]

    return x_train_MF, x_test_NF_new

    '''
    Returns
    -------
    x_train_MF : 2D np.array, shape = (n_train_examples, n_features) (M, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    x_test_NF : 2D np.array, shape = (n_test_examples, n_features) (N, F)
        Each row is a feature vector
        Should be a separately allocated array, NOT a view of any input array

    Post Condition
    --------------
    This function should be side-effect free. Provided input array x_all_LF
    should not change at all (not be shuffled, etc.)

    Examples
    --------


    References
    ----------
    For more about RandomState, see:
    https://stackoverflow.com/questions/28064634/random-state-pseudo-random-numberin-scikit-learn
    '''

# x_LF = np.eye(10)
# xcopy_LF = x_LF.copy() # preserve what input was before the call
# train_MF, test_NF = split_into_train_and_test(x_LF, frac_test=0.201, random_state=np.random.RandomState(0))
# train_MF.shape
# # (7, 10)
# test_NF.shape
# # (3, 10)
# print(train_MF)
# # [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
# #  [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
# #  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 1. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]]
# print(test_NF)
# # [[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
# #  [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
# #  [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]]
#
# ## Verify that input array did not change due to function call
# np.allclose(x_LF, xcopy_LF)
# # True

def euclidean_distance(v1, v2):
        v1,v2 =np.array(v1), np.array(v2)
        distance = 0.0
        for i in range(len(v1)):
            distance +=(v1[i] - v2[i])**2
        return np.sqrt(distance)


def calc_k_nearest_neighbors(data_NF, query_QF, K=1):

    array_3d = np.empty([query_QF.shape[0], K, data_NF.shape[1]])
    for x in range(query_QF.shape[0]):
        print("PROCESSING QUERY:", x)
        new_query_results = np.empty([K, data_NF.shape[1]])
        new_query_distance = np.empty([data_NF.shape[0], 2])
        for i in range(data_NF.shape[0]):
            new_query_distance[i][0] = euclidean_distance(query_QF[x], data_NF[i])
            print("distance", new_query_distance[i][0], "index", i)
            new_query_distance[i][1] = i
        new_query_distance_1 = np.zeros((data_NF.shape[0],2))
        new_query_distance_1 = new_query_distance[new_query_distance[:,0].argsort(kind='mergesort')]
        print(new_query_distance_1)
        for i in range(K):
            array_3d[x][i] = data_NF[int(new_query_distance_1[i,1])]

    return array_3d
    ''' Compute and return k-nearest neighbors under Euclidean distance

    Args
    ----
    data_NF : 2D np.array, shape = (n_examples, n_features) aka (N, F)
        Each row is a feature vector for one example in dataset
    query_QF : 2D np.array, shape = (n_queries, n_features) aka (Q, F)
        Each row is a feature vector whose neighbors we want to find
    K : int, must satisfy K >= 1 and K <= n_examples aka N
        Number of neighbors to find per query vector

    Returns
    -------
    neighb_QKF : 3D np.array, (n_queries, n_neighbors, n_feats) (Q, K, F)
        Entry q,k is feature vector of k-th nearest neighbor of the q-th query
        If two vectors are equally close, then we break ties by taking the one
        appearing first in row order in the original data_NF array

    Examples
    --------

    '''
# data_NF = np.vstack([ np.eye(4), 1.4 * np.eye(4), 0.8 * np.eye(4)])
# query_QF = np.asarray([[0, 0, 0, 1.0]])
# print(calc_k_nearest_neighbors(data_NF, query_QF, 3))
# print(data_NF)


# data_NF = np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]])
# query_QF = np.asarray([[0.9, 0], [0, -0.9]])
# neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=1)
# print(neighb_QKF[0])
# # array([[1., 0.]])
# print(neighb_QKF[1])
# .array([[ 0., -1.]])

#     # Find 3 nearest neighbors
# neighb_QKF = calc_k_nearest_neighbors(data_NF, query_QF, K=3)
# print(neighb_QKF.shape())
# # shape(2, 3, 2)
# print(neighb_QKF[0])
# # array([[ 1.,  0.], [ 0.,  1.], [ 0., -1.]])
# print(neighb_QKF[1])
# # array([[ 0., -1.], [ 1.,  0.], [-1.,  0.]]))
