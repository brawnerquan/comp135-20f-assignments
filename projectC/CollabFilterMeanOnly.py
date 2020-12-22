'''
CollabFilterMeanOnly.py

Defines class: `CollabFilterMeanOnly`

Scroll down to __main__ to see a usage example.

'''

# Make sure you use the autograd version of numpy (which we named 'ag_np')
# to do all the loss calculations, since automatic gradients are needed
import autograd.numpy as ag_np

# Use helper packages
from AbstractBaseCollabFilterSGD import AbstractBaseCollabFilterSGD
from train_valid_test_loader import load_train_valid_test_datasets

# Some packages you might need (uncomment as necessary)
import pandas as pd
import matplotlib.pyplot as plt

# No other imports specific to ML (e.g. scikit) needed!

class CollabFilterMeanOnly(AbstractBaseCollabFilterSGD):
    ''' Simple baseline recommendation model.

    Always predicts same scalar no matter what user/movie.

    Attributes required in param_dict
    ---------------------------------
    mu : 1D array of size (1,)

    Notes
    -----
    Inherits *__init__** constructor from AbstractBaseCollabFilterSGD.
    Inherits *fit* method from AbstractBaseCollabFilterSGD.
    '''

    def init_parameter_dict(self, n_users, n_items, train_tuple):
        ''' Initialize parameter dictionary attribute for this instance.

        Post Condition
        --------------
        Updates the following attributes of this instance:
        * param_dict : dict
            Keys are string names of parameters (e.g. 'mu')
            Values are *numpy arrays* of parameter values
        '''
        # DONE for you. No need to edit.
        # Just initialize to zero always.
        self.param_dict = dict(mu=ag_np.zeros(1))

    def predict(self, user_id_N, item_id_N, mu=None):
        ''' Predict ratings at specific user_id, item_id pairs

        Args
        ----
        user_id_N : 1D array, size n_examples
            Specific user_id values to use to make predictions
        item_id_N : 1D array, size n_examples
            Specific item_id values to use to make predictions
            Each entry is paired with the corresponding entry of user_id_N

        Returns
        -------
        yhat_N : 1D array, size n_examples
            Scalar predicted ratings, one per provided example.
            Entry n is for the n-th pair of user_id, item_id values provided.
        '''
        # TODO: Update with actual prediction logic
        N = user_id_N.size
        if mu is None:
            yhat_N = ag_np.ones(N) * self.param_dict["mu"]
        else:
            yhat_N = ag_np.ones(N) * ag_np.array(mu)
        return yhat_N

    def calc_loss_wrt_parameter_dict(self, param_dict, data_tuple):
        ''' Compute loss at given parameters

        Args
        ----
        param_dict : dict
            Keys are string names of parameters
            Values are *numpy arrays* of parameter values

        Returns
        -------
        loss : float scalar
        '''
        # TODO compute loss
        y_N = data_tuple[2]
        yhat_N = self.predict(data_tuple[0], data_tuple[1], **param_dict)
        
        loss_total = ag_np.sum(  ( ag_np.array(y_N) - yhat_N ) **2 )
        
        return loss_total




if __name__ == '__main__':

    # Load the dataset
    train_tuple, valid_tuple, test_tuple, n_users, n_items = \
        load_train_valid_test_datasets()
    # Create the model and initialize its parameters
    # to have right scale as the dataset (right num users and items)
    model = CollabFilterMeanOnly(
        n_epochs=10, batch_size=100, step_size=0.1)
    model.init_parameter_dict(n_users, n_items, train_tuple)

    # Fit the model with SGD
    model.fit(train_tuple, valid_tuple)

    plt.plot(model.trace_epoch, model.trace_mae_train, c='r', label="training")
    plt.plot(model.trace_epoch, model.trace_mae_valid, c='b', label="validation")
    plt.legend()

    plt.title("Mean Absolute Error V.S. Epochs using 10000 batch size for Model M1: Predict Mean Only")

    idx100 = 19
    idx10k = 14
    
    idx = idx10k
    y_idx = max( max(model.trace_mae_train[idx:]), max(model.trace_mae_valid[idx:]) )
    y_min = min( min(model.trace_mae_train[idx:]), min(model.trace_mae_valid[idx:]) )

    print(y_min, y_idx)
    
    plt.ylim((y_min, y_idx))

    plt.xlabel("Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.show()
    
    import numpy as np
    print(np.mean(train_tuple[2]))
    print(model.param_dict['mu'])


    
