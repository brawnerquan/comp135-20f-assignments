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


def show_images(X, y, row_ids, n_rows=3, n_cols=3):
    ''' Display images

    Args
    ----
    X : 2D array, shape (N, 784)
        Each row is a flat image vector for one example
    y : 1D array, shape (N,)
        Each row is label for one example
    row_ids : list of int
        Which rows of the dataset you want to display
    '''
    fig, axes = plt.subplots(
            nrows=n_rows, ncols=n_cols,
            figsize=(n_cols * 3, n_rows * 3))

    for ii, row_id in enumerate(row_ids):
        cur_ax = axes.flatten()[ii]
        cur_ax.imshow(X[row_id].reshape(28,28), interpolation='nearest', vmin=0, vmax=1, cmap='gray')
        cur_ax.set_xticks([])
        cur_ax.set_yticks([])
        cur_ax.set_title('y=%d' % y[row_id])


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
x_train_old = x_train.copy()
x_train = np.hstack([trans_train, trans_train_T])

print("it took forever")

"""

C = np.logspace(-8, 0, 31)
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

plt.plot(C, C_lst_train, '.-', c='b', label="training set")
plt.plot(C, C_lst_valid, '.-', c='r', label="validation set")

plt.title("Log Loss vs C Hyperparameter Selection")


plt.ylabel("log loss")
plt.xlabel("C")

plt.legend(loc='upper right')
plt.show()

save = best.predict(x_train)
with open("temp.npy", 'wb') as f:
  np.save(f, save)
"""

#save = np.load("temp.npy")

#print(save.shape)



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

save = model.predict_proba(test_in)[:, 1]
fpr, tpr, thresholds = sklearn.metrics.roc_curve(test_out.flatten(), save.flatten())

print(fpr)

plt.plot(fpr, tpr, '.-', c='b', label="validation set")

save = model.predict_proba(train_in)[:, 1]

fpr, tpr, thresholds = sklearn.metrics.roc_curve(train_out.flatten(), save.flatten())

plt.plot(fpr, tpr,'.-', c='r', label="training set")
plt.title("ROC curve of Number of Continuous Strips Model")


plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")

plt.legend(loc='lower right')
plt.show()

y_hat = model.predict(test_in)

idx = list(np.where(np.logical_and(y_hat.flatten() == 1, test_out.flatten() == 0))[0][:3])
idx += list(np.where(np.logical_and(y_hat.flatten() == 0, test_out.flatten() == 1))[0][:3])



print(idx)

true_idx = idx#indexes[idx]

show_images(x_train_old[indexes[:test_size]], y_hat, true_idx, n_rows=2, n_cols=3)

plt.show()



"""

print(len(np.where(save == test_out.flatten())[0]))

idx = np.where(save != test_out.flatten())[0]
print(idx.shape)

for id_ in idx:
    print(test_in[id_])
    plt.imshow(ori_x_train[indexes[id_]].reshape((28,28)))
    plt.show()
"""

    










