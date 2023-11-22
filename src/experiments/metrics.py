# -*- coding: utf-8 -*-
# Author: Li Qingquan
# 评测指标计算
# based on LeCaRD

import os
import numpy as np
import json
import math

def kappa(testData, k): #testData表示要计算的数据，k表示数据矩阵的是k*k的
    dataMat = np.mat(testData)
    P0 = 0.0
    for i in range(k):
        P0 += dataMat[i, i]*1.0
    xsum = np.sum(dataMat, axis=1)
    ysum = np.sum(dataMat, axis=0)
    #xsum是个k行1列的向量，ysum是个1行k列的向量
    Pe  = float(ysum*xsum)/k**2
    P0 = float(P0/k*1.0)
    cohens_coefficient = float((P0-Pe)/(1-Pe))
    return cohens_coefficient

def fleiss_kappa(testData, N, k, n): 
    dataMat = np.mat(testData, float)
    oneMat = np.ones((k, 1))
    sum = 0.0
    P0 = 0.0
    for i in range(N):
        temp = 0.0
        for j in range(k):
            sum += dataMat[i, j]
            temp += 1.0*dataMat[i, j]**2
        temp -= n
        temp /= (n-1)*n
        P0 += temp
    P0 = 1.0*P0/N
    ysum = np.sum(dataMat, axis=0)
    for i in range(k):
        ysum[0, i] = (ysum[0, i]/sum)**2 # (1/k)**2
    Pe = ysum*oneMat*1.0
    ans = (P0-Pe)/(1-Pe)
    return ans[0, 0]

def ndcg(ranks, gt_ranks, K):
    dcg_value = 0.
    idcg_value = 0.
    # log_ki = []

    sranks = sorted(gt_ranks, reverse=True)

    for i in range(0,K):
        logi = math.log(i+2,2)
        dcg_value += ranks[i] / logi
        idcg_value += sranks[i] / logi

    return dcg_value/idcg_value

label_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/labels/top30_dict.json'
label_dict = json.load(open(label_path, 'r'))

train_test_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/train_test.json'
train_test_file = open(train_test_path, 'r')
train_test = json.load(train_test_file)
mode = 'test'  # change mode for different test set
if mode == 'all':
    test_querys = train_test['train'] + train_test['test']
else:
    test_querys = train_test['test']
print(f'Mode: {mode}')

models = ['tfidf', 'bm25', 'lmir', 'labels', 'lfm']  # lfm only for test

for model in models:
    print(f'{model}:')
    pred_path = os.path.join('/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/predictions', model + '_top100.json')
    pred_dict = json.load(open(pred_path, 'r'))

    # ndcg
    print('ndcg')
    topk_list = [10, 20, 30]
    ndcg_list = []
    for topk in topk_list:
        temk_list = []
        sndcg = 0.0
        for key in test_querys:
            rawranks = []
            for i in pred_dict[key]:
                if i in list(label_dict[key])[:30]:
                    rawranks.append(label_dict[key][i])
                else:
                    rawranks.append(0)
            ranks = rawranks + [0] * (30 - len(rawranks))
            if sum(ranks) != 0:
                sndcg += ndcg(ranks, list(label_dict[key].values()), topk)
        temk_list.append(sndcg / len(test_querys))
        ndcg_list.append((topk, temk_list))
    print(ndcg_list)

    # P
    print('P')
    topk_list = [5, 10]
    sp_list = []
    for topk in topk_list:
        temk_list = []
        sp = 0.0
        for key in test_querys:
            ranks = [i for i in pred_dict[key][:topk] if i in list(label_dict[key].keys())[:30]]
            sp += float(len([j for j in ranks[:topk] if label_dict[key][j] >= 5]) / topk)
        temk_list.append(sp / len(test_querys))
        sp_list.append((topk, temk_list))
    print(sp_list)

    # MAP
    print('MAP')
    map_list = []
    smap = 0.0
    for key in test_querys:
        ranks = [i for i in pred_dict[key] if i in label_dict[key]]
        rels = [ranks.index(i) for i in ranks if label_dict[key][i] >= 5]
        tem_map = 0.0
        for rel_rank in rels:
            tem_map += float(len([j for j in ranks[:rel_rank + 1] if label_dict[key][j] >= 5]) / (rel_rank + 1))
        if len(rels) > 0:
            smap += tem_map / len(rels)
    map_list.append(smap / len(test_querys))
    print(map_list)
