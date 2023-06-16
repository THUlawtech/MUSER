import enum
import json
import random
import torch
import os
import numpy as np

from transformers import AutoTokenizer
from Basic import BasicFormatter
from tqdm import tqdm


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    # print(len(tokens_a), len(tokens_b))
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class MuserFormatter(BasicFormatter):
    def __init__(self, plm_path='thunlp/Lawformer', mode='train', query_len=600, cand_len=600):
        super().__init__(plm_path, mode, query_len, cand_len)

        self.tokenizer = AutoTokenizer.from_pretrained(plm_path)
        self.mode = mode
        self.query_len = query_len
        self.cand_len = cand_len
        self.max_len = self.query_len + self.cand_len + 4
        self.get_gat = self.QueryAtt
    
    def QueryAtt(self, query, input_ids):
        ret = [1] * (len(query) + 2)
        ret += [0] * (len(input_ids) - len(ret))
        return ret

    def process(self, data, mode='train'):
        inputx = []
        segment = []
        mask = []
        labels = []
        global_att = []
        for temp in data:
            query = self.tokenizer.tokenize(temp["query"])[:self.query_len]
            cand = self.tokenizer.tokenize(temp["cand"])[:self.cand_len]

            tokens = ["[CLS]"] + query + ["[SEP]"] + cand + ["[SEP]"]
            segment_ids = [0] * (len(query) + 2) + [1] * (len(cand) + 1)

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            gat_mask = self.get_gat(query, input_ids)
            input_mask = [1] * len(input_ids)
            # gat_mask = [1] * (len(query) + 2)

            padding = [0] * (self.max_len - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            gat_mask += padding
            # gat_mask += [0] * (self.max_len - len(gat_mask))

            assert len(input_ids) == self.max_len
            assert len(input_mask) == self.max_len
            assert len(segment_ids) == self.max_len
            assert len(gat_mask) == self.max_len

            inputx.append(input_ids)
            segment.append(segment_ids)
            mask.append(input_mask)
            labels.append(int(temp["label"]))
            global_att.append(gat_mask)

        #global_att = np.zeros((len(data), self.max_len), dtype=np.int32)
        #global_att[:,0] = 1
        return {
            "inputx": torch.LongTensor(inputx),
            "segment": torch.LongTensor(segment),
            "mask": torch.LongTensor(mask),
            "global_att": torch.LongTensor(global_att),
            "labels": torch.LongTensor(labels),
            "index": [temp["index"] for temp in data]
        }

