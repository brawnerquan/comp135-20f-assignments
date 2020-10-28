import numpy as np
import scipy.stats
import pandas as pd
import sklearn.neural_network
# import plotting libraries
import matplotlib
import matplotlib.pyplot as plt

%matplotlib inline
plt.style.use('seaborn') # pretty matplotlib plots

import seaborn as sns
sns.set('notebook', font_scale=1.25, style='whitegrid')

df_x_test = pd.read_csv("x_test.csv")
df_x_train = pd.read_csv("x_train.csv")
df_y_train = pd.read_csv("y_train.csv")


x_test = df_x_test.to_numpy()
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()



ind = np.arange(x_train.shape[0])
np.random.shuffle(ind)
split_size = x_train.shape[0]//2
x_valid = x_train[ind[: split_size]]
y_valid = y_train[ind[: split_size]]
x_train = x_train[ind[split_size:]]
y_train = y_train[ind[split_size:]]

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)

mlp = sklearn.neural_network.MLPClassifier(
    hidden_layer_sizes=[32],
    solver='adam',
    max_iter=1000)

# We can create a *distribution* object using scipy.stats

dist = scipy.stats.randint(4, 64)

my_parameter_distributions_by_name = dict(
    hidden_layer_sizes=scipy.stats.randint(2, 70),
    alpha=scipy.stats.uniform(0.0, 1.0),
    random_state=[  # try two possible seeds to initialize parameters
        101, 202,
        ],
    )

n_trials_rand_search = 12

xall_N2 = np.vstack([x_train, x_valid])
yall_N = np.vstack([y_train, y_valid])

print(xall_N2.shape)
print(yall_N.shape)


valid_indicators_N = np.hstack([
    -1 * np.ones(y_train.size), # -1 means never include this example in any test split
    0  * np.ones(y_valid.size), #  0 means include in the first test split (we count starting at 0 in python)
    ])

# Create splitter object using Predefined Split

my_splitter = sklearn.model_selection.PredefinedSplit(valid_indicators_N)

my_rand_searcher = sklearn.model_selection.RandomizedSearchCV(
    mlp,
    my_parameter_distributions_by_name,
    scoring="accuracy",
    cv=my_splitter,
    n_iter=n_trials_rand_search,
    random_state=101, # same seed means same results everytime we repeat this code
    )

my_rand_searcher.fit(xall_N2, yall_N.flatten())

param_keys = ['param_hidden_layer_sizes', 'param_alpha', 'param_random_state']

# Rearrange row order so it is easy to skim
rsearch_results_df = pd.DataFrame(my_rand_searcher.cv_results_).copy()
rsearch_results_df.sort_values(param_keys, inplace=True)


print("Dataframe has shape: %s" % (str(rsearch_results_df.shape)))

print("Dataframe has columns:")
for c in rsearch_results_df.columns:
    print("-- %s" % c)

rsearch_results_df[param_keys + ['split0_test_score', 'rank_test_score']]


#0.9748333333333333
# MLPClassifier(activation='relu', alpha=0.2323536618147607, batch_size='auto',
#               beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
#               hidden_layer_sizes=61, learning_rate='constant',
#               learning_rate_init=0.001, max_fun=15000, max_iter=1000,
#               momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
#               power_t=0.5, random_state=101, shuffle=True, solver='adam',
#               tol=0.0001, validation_fraction=0.1, verbose=False,
#               warm_start=False)
