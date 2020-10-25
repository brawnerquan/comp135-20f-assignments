import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv("data_sneaker_vs_sandal/x_train.csv").to_numpy()
gt = pd.read_csv("data_sneaker_vs_sandal/y_train.csv").to_numpy()

id1 = np.where(gt.flatten() == 1)
id0 = np.where(gt.flatten() == 0)

s1 = train[id0].flatten()
s2 = train[id1].flatten()

plt.hist([s1[s1>0], s2[s2>0]], 256)
plt.show()

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
