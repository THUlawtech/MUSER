# -*- coding: utf-8 -*-
# Author: Li Qingquan
# 标签匹配类案

import os
import json
import math
from tqdm import tqdm
from scipy import spatial

in_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cases_pool.json'
out_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/predictions/labels_top100.json'

in_file = open(in_path, 'r', encoding='utf-8')
cases_pool = json.load(in_file)

train_test_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/train_test.json'
train_test_file = open(train_test_path, 'r')
train_test = json.load(train_test_file)
test_querys = train_test['train'] + train_test['test']

legal_items_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/legal_items'
aqbq_label2id = json.load(open(os.path.join(legal_items_path, 'aqbq.json'), 'r', encoding='utf-8'))
aqbq_tfidf = json.load(open(os.path.join(legal_items_path, 'aqbq_tfidf.json'), 'r', encoding='utf-8'))
zyjd_label2id = json.load(open(os.path.join(legal_items_path, 'zyjd.json'), 'r', encoding='utf-8'))
zyjd_tfidf = json.load(open(os.path.join(legal_items_path, 'zyjd_tfidf.json'), 'r', encoding='utf-8'))
ft_label2id = json.load(open(os.path.join(legal_items_path, 'ft.json'), 'r', encoding='utf-8'))
ft_tfidf = json.load(open(os.path.join(legal_items_path, 'ft_tfidf.json'), 'r', encoding='utf-8'))

qc_pairs_path = '/Users/qingquanli/Documents/Seafile/CIKM_mjjd/data/cases/cands_by_query.json'
qc_pairs_file = open(qc_pairs_path, 'r')
qc_pairs = json.load(qc_pairs_file)

aqbq_l1_w = len(aqbq_label2id['level1']) / (len(aqbq_label2id['level1']) + len(aqbq_label2id['level2']) + len(aqbq_label2id['level3']))
aqbq_l2_w = len(aqbq_label2id['level2']) / (len(aqbq_label2id['level1']) + len(aqbq_label2id['level2']) + len(aqbq_label2id['level3']))
aqbq_l3_w = len(aqbq_label2id['level3']) / (len(aqbq_label2id['level1']) + len(aqbq_label2id['level2']) + len(aqbq_label2id['level3']))
zyjd_l1_w = len(zyjd_label2id['level1']) / (len(zyjd_label2id['level1']) + len(zyjd_label2id['level2']) + len(zyjd_label2id['level3']))
zyjd_l2_w = len(zyjd_label2id['level2']) / (len(zyjd_label2id['level1']) + len(zyjd_label2id['level2']) + len(zyjd_label2id['level3']))
zyjd_l3_w = len(zyjd_label2id['level3']) / (len(zyjd_label2id['level1']) + len(zyjd_label2id['level2']) + len(zyjd_label2id['level3']))
ft_l1_w = len(ft_label2id['level1']) / (len(ft_label2id['level1']) + len(ft_label2id['level2']))
ft_l2_w = len(ft_label2id['level2']) / (len(ft_label2id['level1']) + len(ft_label2id['level2']))


def get_vectors(case):
    vectors = {
        'aqbq': {
            'level1': [.0] * len(aqbq_label2id['level1']),
            'level2': [.0] * len(aqbq_label2id['level2']),
            'level3': [.0] * len(aqbq_label2id['level3'])
        },
        'zyjd': {
            'level1': [.0] * len(zyjd_label2id['level1']),
            'level2': [.0] * len(zyjd_label2id['level2']),
            'level3': [.0] * len(zyjd_label2id['level3'])
        },
        'ft': {
            'level1': [.0] * len(ft_label2id['level1']),
            'level2': [.0] * len(ft_label2id['level2'])
        }
    }

    for level in ['level1', 'level2', 'level3']:
        for type in ['aqbq', 'zyjd']:
            for label in set(case['split_labels'][type][level]):
                if type == 'aqbq':
                    # vectors[type][level][aqbq_label2id[level][label]] = aqbq_tfidf[level][label]
                    vectors[type][level][aqbq_label2id[level][label]] = 1
                else:
                    # vectors[type][level][zyjd_label2id[level][label]] = zyjd_tfidf[level][label]
                    vectors[type][level][zyjd_label2id[level][label]] = 1
        if level == 'level3':
            break
        for ft in case['content']['法条']:
            comb_ft = ft['法律'] + '-' + ft['序号']
            if comb_ft not in ft_label2id['level2']:
                continue
            # vectors['ft']['level1'][ft_label2id['level1'][ft['法律']]] = ft_tfidf['level1'][ft['法律']]
            # vectors['ft']['level2'][ft_label2id['level2'][comb_ft]] = ft_tfidf['level2'][comb_ft]
            vectors['ft']['level1'][ft_label2id['level1'][ft['法律']]] = 1
            vectors['ft']['level2'][ft_label2id['level2'][comb_ft]] = 1

    return vectors

def cos_sim(list1, list2):
    cos = 1.0 - spatial.distance.cosine(list1, list2)
    if math.isnan(cos):
        return 0.0
    return cos

def get_sim_score(vector1, vector2):
    aqbq_sim_score = aqbq_l1_w * cos_sim(vector1['aqbq']['level1'], vector2['aqbq']['level1']) + aqbq_l2_w * cos_sim(vector1['aqbq']['level2'], vector2['aqbq']['level2']) + aqbq_l3_w * cos_sim(vector1['aqbq']['level3'], vector2['aqbq']['level3'])
    zyjd_sim_score = zyjd_l1_w * cos_sim(vector1['zyjd']['level1'], vector2['zyjd']['level1']) + zyjd_l2_w * cos_sim(vector1['zyjd']['level2'], vector2['zyjd']['level2']) + zyjd_l3_w * cos_sim(vector1['zyjd']['level3'], vector2['zyjd']['level3'])
    ft_sim_score = ft_l1_w * cos_sim(vector1['ft']['level1'], vector2['ft']['level1']) + ft_l2_w * cos_sim(vector1['ft']['level2'], vector2['ft']['level2'])
    # aqbq_sim_score = cos_sim(vector1['aqbq']['level3'], vector2['aqbq']['level3'])
    # zyjd_sim_score = cos_sim(vector1['zyjd']['level3'], vector2['zyjd']['level3'])
    # ft_sim_score = cos_sim(vector1['ft']['level2'], vector2['ft']['level2'])

    sim_score = 0.5 * aqbq_sim_score + 0.4 * zyjd_sim_score + 0.1 * ft_sim_score
    return sim_score

labels_top100 = {}

for qid in tqdm(test_querys):
    sim_scores = []
    query_vectors = get_vectors(cases_pool[qid])
    # print(qid)
    for cid, case in cases_pool.items():
        # if int(cid) == qid or int(cid) not in qc_pairs[qid]:
        if int(cid) == int(qid):
            continue
        cand_vectors = get_vectors(case)
        sim_scores.append((cid, get_sim_score(query_vectors, cand_vectors)))
    # print(qc_pairs[qid])
    # assert len(sim_scores) == 100
    sim_scores.sort(key=lambda x:x[1], reverse=True)
    cnt = 0
    labels_top100[qid] = []
    for idx, score in sim_scores:
        if cnt >= 100:
            break
        labels_top100[qid].append(idx)
        cnt += 1
    assert len(labels_top100[qid]) == 100

out_file = open(out_path, 'w')
json.dump(labels_top100, out_file)
