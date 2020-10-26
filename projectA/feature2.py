import numpy as np
import pandas as pd
import sklearn.linear_model

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from my_filter import convolve

import matplotlib.pyplot as plt

df_x_test = pd.read_csv("data_sneaker_vs_sandal/x_test.csv")
df_x_train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv")
df_y_train = pd.read_csv("data_sneaker_vs_sandal/y_train.csv")


x_test = df_x_test.to_numpy()
x_train = df_x_train.to_numpy()
y_train = df_y_train.to_numpy()


"""
new_x_train = []

train_xs = np.repeat([np.arange(28)], 28, axis=0)
train_ys = train_xs.T

for row in x_train:
    x_pos = (row.reshape((28,28))*train_xs).sum()/(np.sum(row))
    y_pos = (row.reshape((28,28))*train_ys).sum()/(np.sum(row))

    shift_x = int(14 - x_pos) 
    shift_y = int(14 - y_pos)

    ori = row.reshape((28,28)).copy()

    out1 = np.zeros((28,28))

    if shift_x > 0:
        out1[:, shift_x:] = ori[:, :-shift_x]
    elif shift_x < 0:
        out1[:, :shift_x] = ori[:, -shift_x: ]

    out2 = np.zeros((28,28))

    if shift_y > 0:
        out2[shift_y:, :] = out1[:-shift_y, :]
    elif shift_y < 0:
        out2[:shift_y, :] = out1[-shift_y:, : ]

    new_x_train.append(out2.flatten())

train_x = np.array(new_x_train)
    
"""
    

    


def splitter(arr):
    out = []
    for row in arr:
        ans = 0
        for (col, nxt) in zip(row[:-1], row[1:]):
            if col != nxt:
                ans += 1     

        out.append(ans)

    return np.array(out)




trans_train = []
trans_train_T = []

for row in x_train:
    curr = np.where(row > 0, 1, 0)
    curr = splitter(row.reshape((28,28)))
    

    trans_train.append(curr.flatten())

trans_train = np.array(trans_train)

for row in x_train:
    curr = np.where(row > 0, 1, 0)
    curr = splitter(row.reshape((28,28)).T)
    

    trans_train_T.append(curr.flatten())

trans_train_T = np.array(trans_train_T)


ori_x_train = x_train.copy()
x_train = np.hstack([trans_train, trans_train_T])

print("it took forever")


"""
C = np.logspace(-8, 8, 31)
kf = KFold(n_splits=10)

C_lst_train = []
C_lst_valid = []

model_list = []



for c in C:
  model = sklearn.linear_model.LogisticRegression(C=c, solver='sag', max_iter=100)

  avg_score_tr = []
  avg_score_va = []
  for (i, (tr_i, ts_i)) in enumerate(kf.split(x_train)):
      curr_train_x , curr_valid_x = x_train[tr_i], x_train[ts_i]
      curr_train_y , curr_valid_y = y_train[tr_i], y_train[ts_i]
      
      
      model.fit(curr_train_x, curr_train_y.flatten())

      model_list.append(model)     
      
      
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

best = model_list[np.argmin(C_lst_valid)]

print(C[np.argmin(C_lst_valid)])

save = best.predict(x_train)
with open("temp.npy", 'wb') as f:
  np.save(f, save)


#save = np.load("temp.npy")

print(save.shape)

"""

# 0.002154434690031882 -- 2 directions (at 100 iter)
# 0.292 -- 1 directions



#train_in, test_in, train_out, test_out = train_test_split(x_train, y_train, test_size=0.2)

test_size = int(0.2*1200)

indexes = np.arange(1200)
np.random.shuffle(indexes)

train_in = x_train[indexes[test_size:]]
train_out = y_train[indexes[test_size:]]

test_in = x_train[indexes[:test_size]]
test_out = y_train[indexes[:test_size]]



model = sklearn.linear_model.LogisticRegression(C=0.0021, solver='sag', max_iter=3000)
model.fit(train_in, train_out.flatten())

plt.imshow(model.coef_.reshape((28,2)))
plt.show()

save = model.predict(test_in)

print(len(np.where(save == test_out.flatten())[0]))

idx = np.where(save != test_out.flatten())[0]
print(idx.shape)

for id_ in idx:
    print(test_in[id_])
    plt.imshow(ori_x_train[indexes[id_]].reshape((28,28)))
    plt.show()
    










