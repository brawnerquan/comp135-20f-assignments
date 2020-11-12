import numpy as np
import pandas as pd

import xgboost as xgb

import string

from collections import OrderedDict

df_x_train = pd.read_csv("data_reviews/x_train.csv")
df_x_test = pd.read_csv("data_reviews/x_test.csv")
df_y_train = pd.read_csv("data_reviews/y_train.csv")

x_train = df_x_train["text"].values.tolist()
x_test = df_x_test["text"].values.tolist()
y_train = df_y_train.to_numpy().flatten().tolist()




def preprocess(lst):
    return list([entry.lower().translate(str.maketrans('', '', string.punctuation))  for entry in lst])

def tokenize(lst):
    return [entry.split() for entry in lst]

def get_volcab(lst):
    volcab = OrderedDict()

    for line in lst:
        for word in line:

            if word in volcab:
                volcab[word] += 1
            else:
                volcab[word] = 1

    return volcab

def make_index_map(volcab):
    keys = volcab.keys()
    res = {}
    for i, key in enumerate(keys):
        res[key] = i

    return res

    
    

def one_hot(sent, volcab, index_map):
    res = [0]*len(volcab.keys())
    for word in sent:
        res[index_map[word]] = 1

    return res

def count_list(sent, volcab, index_map):
    res = [0]*len(volcab.keys())
    for word in sent:
        if word not in index_map:
            continue
        res[index_map[word]] += 1

    return res

    
    

def create_vector(volcab, lst, func):
    index_map = make_index_map(volcab)
    return np.array([func(entry, volcab, index_map) for entry in lst])

data_train = tokenize(preprocess(x_train))
data_test = tokenize(preprocess(x_test))
volcab = get_volcab(data_train)

res_train = create_vector(volcab, data_train, count_list)
res_test = create_vector(volcab, data_test, count_list)

print(res_train.shape)
print(res_test.shape)

df_out = pd.DataFrame(res_train)
df_out.to_csv("bag_of_words/pre_train.csv", header=False)

df_out = pd.DataFrame(res_test)
df_out.to_csv("bag_of_words/pre_test.csv", header=False)





