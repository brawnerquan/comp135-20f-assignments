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

def transform(in_data):

    k = np.ones((3,3))

    trans_train = convolve2(np.where(in_data > 0, 1, 0).reshape(-1, 28, 28), 28, 28, k)
    trans_train = np.where(np.logical_and(trans_train > 0, trans_train < 9), 1, 0)
    trans_train = trans_train.reshape((-1, 28**2))


    trans_train = np.array(trans_train)

    trans_train2 = convolve2(np.where(in_data > 0, 1, 0).reshape(-1, 28, 28), 28, 28, k)
    trans_train2 = trans_train2.reshape((-1, 28**2))
    return np.hstack([in_data, trans_train, trans_train2])

x_train = transform(x_train)
x_test = transform(x_test)

print("it took forever")

model = sklearn.linear_model.LogisticRegression(C=0.0073564225, solver='sag', max_iter=3000)


model.fit(x_train, y_train.flatten())



#pl.imshow(model.coef_.reshape((28,28)))
#plt.show()

prob = model.predict_proba(x_test)
print(prob)
np.savetxt("y_test.txt", prob[:, 1].flatten())
