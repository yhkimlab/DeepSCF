from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import glob
import pickle
import h5py
import hydra
import os, sys

path = sys.argv[1]


with open(path,'rb') as f:
    data_list = pickle.load(f)

# dataset
train_set = data_list['train']
test_set = data_list['test']

print(train_set)

train_set_new = []
test_set_new = []

origin = '/home2/rong/01.Research/3.Projects/2.ML/0.Clean_up/datasets/no_augmentation/'

for i in range(len(train_set)):

    name = train_set[i].split('/')[-1]
    name = origin + name
    train_set_new.append(name)

    print(name)

for i in range(len(test_set)):

    name = test_set[i].split('/')[-1]
    name = origin + name
    print(name)

    test_set_new.append(name)


# save dataset information
dataset_info = {'train': list(train_set_new),
                'test': list(test_set_new)
                }
with open('data_new.pkl', 'wb') as f:
    pickle.dump(dataset_info, f)
