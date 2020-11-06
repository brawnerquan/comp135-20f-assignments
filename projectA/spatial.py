import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from my_filter import convolve2

train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv").to_numpy()
gt = pd.read_csv("data_sneaker_vs_sandal/y_train.csv").to_numpy()

id1 = np.where(gt.flatten() == 1)
id0 = np.where(gt.flatten() == 0)

s1 = train[id0].flatten()
s2 = train[id1].flatten()

"""
plt.hist([s1[s1>0], s2[s2>0]], 256)
plt.show()

"""

"""
#total on pixels have same mean different variance
#but by threshold at 0.5 we get bunny ears of each, major overlap still and huge skew

test = np.where(train > 0.5, 1, 0).sum(axis=1)

plt.hist([test[id1], test[id0]], 256)
plt.show()
"""

"""
# distribution along the rows
test = np.where(train > 0, 1, 0)
def splitter(arr):
    out = []
    for row in arr:
        ans = 0
        for (col, nxt) in zip(row[:-1], row[1:]):
            if col != nxt:
                ans += 1     

        out.append(ans)

    return np.array(out)

res = []

for img in test:
    res.append(splitter(img.reshape((28,28))))

res = np.array(res)

print(res)


plt.hist(res[id0].T)
plt.figure()
plt.hist(res[id1].T)
plt.show()
"""   

"""

x_train = np.repeat([np.arange(28)], 28, axis=0)
y_train = x_train.T

cent_xs = []
cent_ys = []
for row in train:
    x_pos = (row.reshape((28,28))*x_train).sum()/(np.sum(row))
    y_pos = (row.reshape((28,28))*y_train).sum()/(np.sum(row))

    cent_xs.append(x_pos)
    cent_ys.append(y_pos)

plt.hist2d(cent_xs, cent_ys, 100)
plt.show()

"""

"""
train = np.where(train > 0, 1, 0)

k = np.ones((3,3))

train_out = train.reshape((-1, 28, 28))

train_out = convolve2(train_out, 28,28, k)
train_out = np.where(np.logical_and(train_out < 9, train_out > 0), 1, 0)

xs = list([np.where(row > 0)[1] for row in train_out])    
ys = list([np.where(row > 0)[0] for row in train_out])

xs = np.array(xs)
ys = np.array(ys)




x_list = np.concatenate(xs[id1])
y_list = np.concatenate(ys[id1])



plt.hist2d(x_list, y_list, 100)
plt.show()
"""


train = np.where(train > 0, 1, 0)
train_out = train.reshape((-1, 28,28))

train_out = convolve2(train_out, 28, 28, np.ones((3,3))).astype(int)

train_out1 = train_out[id1].mean(axis=0)

train_out1 = train_out1 / np.max(train_out1)

train_out0 = train_out[id0].mean(axis=0)
train_out0 = train_out0 / np.max(train_out0)

plt.imshow(train_out1)
plt.figure()
plt.imshow(train_out0)
plt.show()


"""
train = np.where(train > 0, 1, 0)
train_out = train.reshape((-1, 28,28))

train_out = convolve2(train_out, 28, 28, np.ones((3,3))).astype(int)
train_out = np.where(np.logical_and(train_out < 9, train_out > 0), 1, 0)

train_out1 = train_out[id1].mean(axis=0)

train_out1 = train_out1 / np.max(train_out1)

train_out0 = train_out[id0].mean(axis=0)
train_out0 = train_out0 / np.max(train_out0)

plt.imshow(train_out1)
plt.figure()
plt.imshow(train_out0)
plt.show()
"""


#plt.imshow(train.mean(axis=0).reshape((28,28)))
#plt.show()



        
