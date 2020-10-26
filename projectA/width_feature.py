import numpy as np
import pandas as pd

import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.model_selection import KFold, train_test_split

from my_filter import convolve, convolve2

import matplotlib.pyplot as plt

df_x_test = pd.read_csv("data_sneaker_vs_sandal/x_test.csv")
df_x_train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv")
df_y_train = pd.read_csv("data_sneaker_vs_sandal/y_train.csv")

x_test = df_x_test.to_numpy()
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()

def splitter(img):
    out = []
    for row in img:
        new_row = []
        counts = [0]
        index = []

        for (i, col) in enumerate(row[:-1]):
            counts[-1] += 1
            if col != row[i+1]:
                counts.append(0)
                index.append(i)

        index.append(len(row)-1)
        
        curr_i = 0
        for (i,col) in enumerate(row):
            if i > index[curr_i]:
                curr_i += 1
            
            res = counts[curr_i]
            new_row.append(res)

        out.append(new_row)

    return np.array(out)             


new_train1 = []
new_train2 = []

for row in x_train:
    new_train1.append(splitter(row.reshape((28,28))).flatten())

for row in x_train:
    new_train2.append(splitter(row.reshape((28,28)).T).flatten())

new_train1 = np.array(new_train1)
new_train2 = np.array(new_train2)

x_train = np.hstack([new_train1, new_train2])

print("finally done")

"""
C = np.logspace(-8, 8, 31)
kf = KFold(n_splits=10)

C_lst_train = []
C_lst_valid = []


for c in C:
  model = sklearn.linear_model.LogisticRegression(C=c, solver='lbfgs', max_iter=100)

  avg_score_tr = []
  avg_score_va = []
  for (i, (tr_i, ts_i)) in enumerate(kf.split(x_train)):
      curr_train_x , curr_valid_x = x_train[tr_i], x_train[ts_i]
      curr_train_y , curr_valid_y = y_train[tr_i], y_train[ts_i]
      
      
      model.fit(curr_train_x, curr_train_y.flatten())

      
      
      
      y_hat = model.predict(curr_train_x).flatten()
      log_loss = sklearn.metrics.log_loss(curr_train_y, y_hat)
      avg_score_tr.append(log_loss)
      y_hat = model.predict(curr_valid_x).flatten()
      log_loss = sklearn.metrics.log_loss(curr_valid_y, y_hat)
      avg_score_va.append(log_loss)
      print(i)
  print("yes sir")
  C_lst_train.append(np.mean(avg_score_tr))
  C_lst_valid.append(np.mean(avg_score_va))

print(C_lst_train)
print(C_lst_valid)



print(C[np.argmin(C_lst_valid)])       
"""

#135.93563908785242

train_in, test_in, train_out, test_out = train_test_split(x_train, y_train, test_size=0.5)
model = sklearn.linear_model.LogisticRegression(C=135.9356, solver='sag', max_iter=6000)
model.fit(train_in, train_out.flatten())


save = model.predict(test_in)

print(len(np.where(save == test_out.flatten())[0]))
print(test_out.shape)

from joblib import dump, load

dump(model, "width.joblib")
        
