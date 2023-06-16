# -*- coding: utf-8 -*-
# Author: Li Qingquan
# 生成jieba分词的语料库

import os
import json
import jieba
from tqdm import tqdm

in_path = 'E:/CIKM_mjjd/data/cases/cases_pool.json'
out_path = 'E:/CIKM_mjjd/data/cases/corpus.json'

in_file = open(in_path, 'r', encoding='utf-8')
cases_pool = json.load(in_file)
corpus = []

stopword_path = 'E:/CIKM_mjjd/data/utils/stopword.txt'
stopword_file = open(stopword_path, 'r', encoding='utf-8')
lines = stopword_file.readlines()
stopwords = [i.strip() for i in lines]
stopwords.extend(['.','（','）','-', '', '【', '】'])

for idx, case in tqdm(cases_pool.items()):
    text = ''
    for part in ['本院查明', '本院认为']:
        for sent in case['content'][part]:
            text += sent
    cutted_tmp = jieba.cut(text, cut_all=False)
    cutted_case = ' '.join(cutted_tmp).split()
    corpus.append([w for w in cutted_case if not w in stopwords])

print(len(corpus))
out_file = open(out_path, 'w', encoding='utf-8')
json.dump(corpus, out_file, ensure_ascii=False)
