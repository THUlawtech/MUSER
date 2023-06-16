import json
import os

from torch.utils.data import Dataset


class MuserDataset(Dataset):
    def __init__(self, data_path, mode='train', encoding='utf-8'):
        self.mode = mode

        self.raw_data = json.load(open(data_path, 'r', encoding=encoding))
        self.data = []

        cnt = 0
        for item in self.raw_data:
            self.data.append({
                'query': item['q_bycm'] + item['q_byrw'],
                'cand': item['c_bycm'] + item['c_byrw'],
                'label': item['label'],
                'index': cnt
            })
            cnt += 1

    def __getitem__(self, item):
        return self.data[item % len(self.data)]

    def __len__(self):
        return len(self.data)

