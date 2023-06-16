import json
import torch
from torch.utils.data import TensorDataset, random_split
import random
from tqdm import tqdm
import os


def dfs_search(path, recursive):
    if os.path.isfile(path):
        return [path]
    file_list = []
    name_list = os.listdir(path)
    name_list.sort()
    for filename in name_list:
        real_path = os.path.join(path, filename)

        if os.path.isdir(real_path):
            if recursive:
                file_list = file_list + dfs_search(real_path, recursive)
        else:
            file_list.append(real_path)

    return file_list
