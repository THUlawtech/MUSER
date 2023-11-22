# -*- coding: utf-8 -*-
# Author: Li Qingquan
# bm25匹配类案

import os
import json
import jieba
from tqdm import tqdm
from gensim.summarization import bm25

in_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cases_pool.json'
out_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/predictions/bm25_top100.json'

in_file = open(in_path, 'r', encoding='utf-8')
cases_pool = json.load(in_file)

corpus_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/corpus.json'
corpus_file = open(corpus_path, 'r', encoding='utf-8')
corpus = json.load(corpus_file)

stopword_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/utils/stopword.txt'
stopword_file = open(stopword_path, 'r', encoding='utf-8')
lines = stopword_file.readlines()
stopwords = [i.strip() for i in lines]
stopwords.extend(['.','（','）','-', '', '【', '】'])

train_test_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/train_test.json'
train_test_file = open(train_test_path, 'r')
train_test = json.load(train_test_file)
test_querys = train_test['train'] + train_test['test']  # test all

qc_pairs_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cands_by_query.json'
qc_pairs_file = open(qc_pairs_path, 'r')
qc_pairs = json.load(qc_pairs_file)

bm25Model = bm25.BM25(corpus)

bm25_top100 = {}
for qid in tqdm(test_querys):
    sim_scores = []
    query_text = ''
    for part in ['本院查明', '本院认为']:
        for sent in cases_pool[qid]['content'][part]:
            query_text += sent
    query_jieba = jieba.cut(query_text, cut_all=False)
    query_tmp = ' '.join(query_jieba).split()
    query_cutted = [w for w in query_tmp if w not in stopwords]
    sim = bm25Model.get_scores(query_cutted)
    # print(sim)
    # print(len(sim))
    # break
    for idx, score in zip(cases_pool.keys(), sim):
        i = int(idx)
        if qid == i or i not in qc_pairs[qid]:
        # if int(qid) == i:
            continue
        sim_scores.append((idx, score))
    # assert len(sim_scores) == 100
    sim_scores.sort(key=lambda x:x[1], reverse=True)
    # print(sim_scores)
    cnt = 0
    bm25_top100[qid] = []
    for idx, score in sim_scores:
        if cnt >= 100:
            break
        bm25_top100[qid].append(idx)
        cnt += 1
    assert len(bm25_top100[qid]) == 100

out_file = open(out_path, 'w')
json.dump(bm25_top100, out_file)
