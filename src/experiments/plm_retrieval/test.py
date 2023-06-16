import os
import json
import math
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from MuserPLM import MuserPLM


root_dir = '/data2/private/liqingquan/mjjd_simcase/plm_retrieval'

cases_pool = json.load(open(os.path.join(root_dir, 'retrieval_test/cases_pool.json'), 'r', encoding='utf-8'))

train_test = json.load(open(os.path.join(root_dir, 'retrieval_test/train_test.json'), 'r'))
test_querys = train_test['test']

qc_pairs = json.load(open(os.path.join(root_dir, 'retrieval_test/cands_by_query.json'), 'r'))

model = MuserPLM()
checkpoint = torch.load(os.path.join(root_dir, 'checkpoints_more/27_checkpoint.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

tokenizer = AutoTokenizer.from_pretrained('thunlp/Lawformer')

max_len = 300

def joint_sents(list1, list2):
    sent1 = ''.join(list1)
    if len(sent1) > max_len:
        sent1 = sent1[:max_len]
    sent2 = ''.join(list2)
    if len(sent2) > max_len:
        sent2 = sent2[:max_len]

    return sent1, sent2

def QueryAtt(query, input_ids):
    ret = [1] * (len(query) + 2)
    ret += [0] * (len(input_ids) - len(ret))
    return ret

def get_input(q_bycm, q_byrw, c_bycm, c_byrw):
    query = tokenizer.tokenize(q_bycm + q_byrw)
    cand = tokenizer.tokenize(c_bycm + c_byrw)

    tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    gat_mask = QueryAtt(query, input_ids)
    input_mask = [1] * len(input_ids)

    return torch.LongTensor([input_ids]), torch.LongTensor([input_mask]), torch.LongTensor([gat_mask])

labels_top100 = {}

for qid in test_querys:
    sim_scores = []
    query = cases_pool[qid]
    q_bycm, q_byrw = joint_sents(query['content']['本院查明'], query['content']['本院认为'])
    for cid, case in tqdm(cases_pool.items(), desc=f'qid: {qid}'):
        # if int(cid) == qid or int(cid) not in qc_pairs[qid]:
        if int(cid) == int(qid):
            continue
        c_bycm, c_byrw = joint_sents(case['content']['本院查明'], case['content']['本院认为'])
        with torch.no_grad():
            input_ids, input_mask, gat_mask = get_input(q_bycm, q_byrw, c_bycm, c_byrw)
            input_ids = input_ids.cuda(non_blocking=True)
            input_mask = input_mask.cuda(non_blocking=True)
            gat_mask = gat_mask.cuda(non_blocking=True)
            output = model(
                data={
                    'inputx': input_ids,
                    'mask': input_mask,
                    'global_att': gat_mask,
                    'labels': torch.LongTensor([0]).cuda(non_blocking=True),
                    'index': torch.LongTensor([0]).cuda(non_blocking=True)
                },
                acc_result=None,
                mode='test'
            )
        sim_scores.append((cid, output['score'][0]))
    # assert len(sim_scores) == 100
    sim_scores.sort(key=lambda x:x[1], reverse=True)
    top_100 = [idx for idx, score in sim_scores]
    labels_top100[qid] = top_100[:100]
    assert len(labels_top100[qid]) == 100

out_file = open(os.path.join(root_dir, 'retrieval_test/lfm_top100_more.json'), 'w')
json.dump(labels_top100, out_file)

