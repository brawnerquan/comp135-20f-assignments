import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.model_selection import KFold

df_x_test = pd.read_csv("data_sneaker_vs_sandal/x_test.csv")
df_x_train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv")
df_y_train = pd.read_csv("data_sneaker_vs_sandal/y_train.csv")


x_test = df_x_test.to_numpy()
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()

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
  C_lst_train.append(np.mean(avg_score_tr))
  C_lst_valid.append(np.mean(avg_score_va))

print(C_lst_train)
print(C_lst_valid)
