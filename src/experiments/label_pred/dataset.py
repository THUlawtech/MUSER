# -*- coding: utf-8 -*-
# @Time : 3/27/21 2:54 PM
# @Author : leoyao
import json
import torch
from torch.utils.data import Dataset


class SentenceData(Dataset):
    def __init__(self, file_name):
        self.data = json.load(open(file_name, encoding='utf-8'))
        # self.label2id = json.load(open('./data/utils/label2id.json', encoding='utf-8'))['level3']
        # self.label2id = json.load(open('./part/label2id.json', encoding='utf-8'))
        self.label2id = json.load(open('./民间借贷6批结果9.2/label2id-final.json', encoding='utf-8'))

    def __getitem__(self, idx):
        sentence = self.data[idx]['sentence']
        labels = self.data[idx]['labels']
        target = torch.zeros(212).long()
        for label in labels:
            target[self.label2id[label]] = 1
        return sentence, target

    def __len__(self):
        return len(self.data)


class SentenceData_ZY(Dataset):
    def __init__(self, file_name):
        self.data = json.load(open(file_name, encoding='utf-8'))
        # self.label2id = json.load(open('./data/utils/label2id.json', encoding='utf-8'))['level3']
        # self.label2id = json.load(open('./part/label2id.json', encoding='utf-8'))
        self.label2id = json.load(open('./民间借贷6批结果9.2/label2id-final-zy.json', encoding='utf-8'))

    def __getitem__(self, idx):
        sentence = self.data[idx]['sentence']
        labels = self.data[idx]['labels']
        target = torch.zeros(295).long()
        for label in labels:
            target[self.label2id[label]] = 1
        return sentence, target

    def __len__(self):
        return len(self.data)