# -*- coding: utf-8 -*-
# Author: Li Qingquan
# tf-idf匹配类案

import os
import json
import jieba
from tqdm import tqdm
from gensim import corpora, models, similarities

in_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cases_pool.json'
out_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/predictions/tfidf_top100.json'

in_file = open(in_path, 'r', encoding='utf-8')
cases_pool = json.load(in_file)

corpus_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/corpus.json'
corpus_file = open(corpus_path, 'r', encoding='utf-8')
raw_corpus = json.load(corpus_file)

stopword_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/utils/stopword.txt'
stopword_file = open(stopword_path, 'r', encoding='utf-8')
lines = stopword_file.readlines()
stopwords = [i.strip() for i in lines]
stopwords.extend(['.','（','）','-', '', '【', '】'])

train_test_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/train_test.json'
train_test_file = open(train_test_path, 'r')
train_test = json.load(train_test_file)
test_querys = train_test['train'] + train_test['test']

qc_pairs_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cands_by_query.json'
qc_pairs_file = open(qc_pairs_path, 'r')
qc_pairs = json.load(qc_pairs_file)

dictionary = corpora.Dictionary(raw_corpus)
corpus = [dictionary.doc2bow(i) for i in raw_corpus]
tfidf = models.TfidfModel(corpus)
num_features = len(dictionary.token2id.keys())
index = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=num_features)

tfidf_top100 = {}
for qid in tqdm(test_querys):
    sim_scores = []
    query_text = ''
    for part in ['本院查明', '本院认为']:
        for sent in cases_pool[qid]['content'][part]:
            query_text += sent
    query_jieba = jieba.cut(query_text, cut_all=False)
    query_tmp = ' '.join(query_jieba).split()
    query_cutted = [w for w in query_tmp if w not in stopwords]
    query_vec = dictionary.doc2bow(query_cutted)
    sim = index[tfidf[query_vec]]
    for idx, score in zip(cases_pool.keys(), sim):
        i = int(idx)
        # if qid == i or i not in qc_pairs[qid]:
        if int(qid) == i:
            continue
        sim_scores.append((idx, score))
    # assert len(sim_scores) == 100
    sim_scores.sort(key=lambda x:x[1], reverse=True)
    # print(sim_scores)
    cnt = 0
    tfidf_top100[qid] = []
    for idx, score in sim_scores:
        if cnt >= 100:
            break
        tfidf_top100[qid].append(idx)
        cnt += 1
    assert len(tfidf_top100[qid]) == 100

out_file = open(out_path, 'w')
json.dump(tfidf_top100, out_file)
