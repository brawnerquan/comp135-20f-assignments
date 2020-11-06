import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

x = np.loadtxt("y_test.txt")
y = np.loadtxt("oldyproba1_test.txt")

x = np.where(x > 0.5, 1, 0)

idx = np.where(x != y)[0]

print(len(idx))

data = pd.read_csv("data_sneaker_vs_sandal/x_test.csv").to_numpy()

for entry in data[idx]:
    img = entry.reshape((28, 28))
    
    plt.imshow(img)
    plt.show()
