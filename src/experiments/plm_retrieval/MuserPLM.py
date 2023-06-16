import torch
import torch.nn as nn
import torch.nn.functional as F
import json

from transformers import AutoConfig, LongformerModel


class MuserPLM(nn.Module):
    def __init__(self, plm_path='thunlp/Lawformer'):
        super(MuserPLM, self).__init__()

        self.encoder = LongformerModel.from_pretrained(plm_path, output_hidden_states=False)
        self.plm_config = AutoConfig.from_pretrained(plm_path)

        self.hidden_size = self.plm_config.hidden_size
        self.fc = nn.Linear(self.hidden_size, 2)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, acc_result, mode):
        inputx = data['inputx']
        out = self.encoder(inputx, attention_mask = data['mask'], global_attention_mask = data["global_att"])
        y = out['pooler_output']
        result = self.fc(y)
        loss = self.criterion(result, data["labels"])
        acc_result = accuracy(result, data["labels"], acc_result)
        if mode == "train":
            return {"loss": loss, "acc_result": acc_result}
            # return {"result": result, 'labels': data["labels"]}
        else:
            score = torch.softmax(result, dim = 1) # batch, 2
            return {"loss": loss, "acc_result": acc_result, "score": score[:,1].tolist(), "index": data["index"]}
            # return {"result": result, "score": y, "index": data["index"]}


def accuracy(logit, label, acc_result):
    if acc_result is None:
        acc_result = {'right': 0, 'actual_num': 0, 'pre_num': 0}
    pred = torch.max(logit, dim = 1)[1]
    acc_result['pre_num'] += int((pred == 1).sum())
    acc_result['actual_num'] += int((label == 1).shape[0])
    acc_result['right'] += int((pred[label == 1] == 1).sum())
    return acc_result

