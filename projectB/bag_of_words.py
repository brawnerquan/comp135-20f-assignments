import pandas as pd
import numpy as np
import xgboost as xgb

df_y_train = pd.read_csv("data_reviews/y_train.csv")

y_train = df_y_train.to_numpy().flatten().tolist()


"""
df_x_train = pd.read_csv("bag_of_words/pre_train.csv")

x_train = df_x_train.to_numpy()

print(len(y_train))
print(x_train.shape)

"""

dtrain = xgb.DMatrix("bag_of_words/pre_train.csv?format=csv", y_train)

param = {

    'max_depth': 2,
    'eta': 1,
    'objective': 'binary:logistic'
}

num_round = 10000

bst = xgb.train(param, dtrain, num_round)

preds = bst.predict(dtrain)

preds = np.array(preds)
y_train = np.array(y_train)

print(len(np.where(y_train == preds)[0])/2400)


