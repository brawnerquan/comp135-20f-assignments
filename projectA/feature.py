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

k = np.ones((3,3))

trans_train = convolve2(np.where(x_train > 0, 1, 0).reshape(-1, 28, 28), 28, 28, k)
trans_train = trans_train.reshape((-1, 28**2))


trans_train = np.array(trans_train)

x_train = trans_train

x_test = convolve2(np.where(x_test > 0, 1, 0).reshape(-1, 28, 28), 28, 28, k).reshape((-1, 28**2))


print("it took forever")

"""
C = np.logspace(-8, 8, 31)
kf = KFold(n_splits=10)

C_lst_train = []
C_lst_valid = []


for c in C:
  model = sklearn.linear_model.LogisticRegression(C=c, solver='lbfgs', max_iter=1000)

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

#0.00018478497974222906
#0.007356422544596407 - 1000 iter training


"""
train_in, test_in, train_out, test_out = train_test_split(x_train, y_train, test_size=0.1)
model = sklearn.linear_model.LogisticRegression(C=0.0073564225, solver='sag', max_iter=3000)
model.fit(train_in, train_out.flatten())


save = model.predict(test_in)

print(len(np.where(save == test_out.flatten())[0]))
print(test_out.shape)
"""

#model = sklearn.linear_model.LogisticRegression(C=0.0021, solver='sag', max_iter=3000)
#model.fit(x_train, y_train.flatten())

#save = model.predict(x_train)

#print(len(np.where(save == y_train.flatten())[0]))
#print(y_train.shape)
#print(sklearn.metrics.zero_one_loss(save, test_out))

id0 = np.where(y_train.flatten() == 0)[0]
id1 = np.where(y_train.flatten() == 1)[0]

model = sklearn.linear_model.LogisticRegression(penalty='l1', C=0.0073564225, solver='saga', max_iter=1)

#model.coef_ = x_train[id1].mean(axis=0) - x_train[id0].mean(axis=0)

model.fit(x_train, y_train.flatten())



plt.imshow(model.coef_.reshape((28,28)))
plt.show()

prob = model.predict_proba(x_test)
print(prob)
np.savetxt("y_test.txt", prob[:, 1].flatten())





