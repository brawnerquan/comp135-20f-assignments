'''
calc_performance_metrics_for_binary_predictions

Provides implementation of common metrics for assessing a binary classifier's
hard decisions against true binary labels, including:
* accuracy
* true positive rate and true negative rate (TPR and TNR)
* positive predictive value and negative predictive value (PPV and NPV)
'''

import numpy as np

def calc_TP_TN_FP_FN(ytrue_N, yhat_N):
    ''' Count the four possible states of true and predicted binary values.

    Args
    ----
    ytrue_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array, shape (n_examples,) = (N,)
        All values must be either 0 or 1. Will be cast to int dtype.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    TP : int
        Number of true positives
    TN : int
        Number of true negatives
    FP : int
        Number of false positives
    FN : int
        Number of false negatives

    Examples
    --------

    '''
    # Cast input to integer just to be sure we're getting what's expected
    ytrue_N = np.asarray(ytrue_N, dtype=np.int32)
    yhat_N = np.asarray(yhat_N, dtype=np.int32)

    # TODO fix by calculating the number of true pos, true neg, etc.
    TP_arr = np.bitwise_and(ytrue_N, yhat_N)
    TN_arr = np.bitwise_and(np.logical_not(ytrue_N), np.logical_not(yhat_N))

    NP = np.count_nonzero(yhat_N == 1)
    NN = np.count_nonzero(yhat_N == 0)

    TP  = np.count_nonzero(TP_arr == 1)
    TN = np.count_nonzero(TN_arr == 1)
    FP = NP - TP
    FN = NN - TN
    return TP, TN, FP, FN


def calc_ACC(ytrue_N, yhat_N):
    ''' Compute the accuracy of provided predicted binary values.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    acc : float
        Accuracy = ratio of number correct over total number of examples

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> acc = calc_ACC(ytrue_N, yhat_N)
    >>> print("%.3f" % acc)
    0.625
    '''

    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N);
    # TODO compute accuracy
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero

    # by adding a small value like 1e-10
    return (TP + TN) / (TP + TN + FN + FP + 1e-10)



def calc_TPR(ytrue_N, yhat_N):
    ''' Compute the true positive rate of provided predicted binary values.

    Also known as the recall.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tpr : float
        TPR = ratio of true positives over total labeled positive

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> tpr = calc_TPR(ytrue_N, yhat_N)
    >>> print("%.3f" % tpr)
    0.500

    # Verify what happens with empty input
    >>> empty_val = calc_TPR([], [])
    >>> print("%.3f" % empty_val)
    0.000
    '''
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N);

    # by adding a small value like 1e-10
    return (TP) / (TP + FN + 1e-10)


def calc_TNR(ytrue_N, yhat_N):
    ''' Compute the true negative rate of provided predicted binary values.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    tnr : float
        TNR = ratio of true negatives over total labeled negative.

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> tnr = calc_TNR(ytrue_N, yhat_N)
    >>> print("%.3f" % tnr)
    0.750

    # Verify what happens with empty input
    >>> empty_val = calc_TNR([], [])
    >>> print("%.3f" % empty_val)
    0.000
    '''
    # TODO compute TNR
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N);

    # by adding a small value like 1e-10
    return (TN) / (FP + TN + 1e-10)



def calc_PPV(ytrue_N, yhat_N):
    ''' Compute positive predictive value of provided predicted binary values.

    Also known as the precision.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    ppv : float
        PPV = ratio of true positives over total predicted positive.

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> ppv = calc_PPV(ytrue_N, yhat_N)
    >>> print("%.3f" % ppv)
    0.667

    # Verify what happens with empty input
    >>> empty_val = calc_PPV([], [])
    >>> print("%.3f" % empty_val)
    0.000
    '''
    # TODO compute PPV
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N);

    return (TP) / (TP + FP + 1e-10)


def calc_NPV(ytrue_N, yhat_N):
    ''' Compute negative predictive value of provided predicted binary values.

    Args
    ----
    ytrue_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents the binary 'true' label of one example
        One entry per example in current dataset
    yhat_N : 1D array of floats, shape (n_examples,) = (N,)
        All values must be either 0.0 or 1.0.
        Each entry represents a predicted label for one example
        One entry per example in current dataset.
        Needs to be same size as ytrue_N.

    Returns
    -------
    npv : float
        NPV = ratio of true negative over total predicted negative.

    Examples
    --------
    >>> N = 8
    >>> ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
    >>> yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
    >>> npv = calc_NPV(ytrue_N, yhat_N)
    >>> print("%.3f" % npv)
    0.600

    # Verify what happens with empty input
    >>> empty_val = calc_NPV([], [])
    >>> print("%.3f" % empty_val)
    0.000
    '''
    # TODO compute NPV
    # You should *use* your calc_TP_TN_FP_FN function from above
    # Hint: make sure denominator will never be exactly zero
    # by adding a small value like 1e-10
    TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N);

    # by adding a small value like 1e-10
    return (TN) / (TN + FN + 1e-10)


# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# TP, TN, FP, FN = calc_TP_TN_FP_FN(ytrue_N, yhat_N)
# print(TP)
# #2
# print(TN)
# #3
# print(FP)
# #1
# print(FN)
# #2
# print(np.allclose(TP + TN + FP + FN, N))
# #True


# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# acc = calc_ACC(ytrue_N, yhat_N)
# print("%.3f" % acc)
# #0.625


# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# tpr = calc_TPR(ytrue_N, yhat_N)
# print("%.3f" % tpr)
# # 0.500
#
# # Verify what happens with empty input
# empty_val = calc_TPR([], [])
# print("%.3f" % empty_val)
# # 0.000
#
# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# tnr = calc_TNR(ytrue_N, yhat_N)
# print("%.3f" % tnr)
# # 0.750
#
# # Verify what happens with empty input
# empty_val = calc_TNR([], [])
# print("%.3f" % empty_val)
# # 0.000


# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# ppv = calc_PPV(ytrue_N, yhat_N)
# print("%.3f" % ppv)
# # 0.667
#
# # Verify what happens with empty input
# empty_val = calc_PPV([], [])
# print("%.3f" % empty_val)
# # 0.000


# N = 8
# ytrue_N = np.asarray([0., 0., 0., 0., 1., 1., 1., 1.])
# yhat_N  = np.asarray([0., 0., 1., 0., 1., 1., 0., 0.])
# npv = calc_NPV(ytrue_N, yhat_N)
# print("%.3f" % npv)
# # 0.600
#
# # Verify what happens with empty input
# empty_val = calc_NPV([], [])
# print("%.3f" % empty_val)
# # 0.000
