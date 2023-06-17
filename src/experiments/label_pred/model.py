import torch.nn as nn
from transformers import BertModel
from LogSumExpLoss import log_sum_exp


class SentenceClassification(nn.Module):
    def __init__(self):
        super(SentenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-chinese')
        self.linear = nn.Linear(in_features=768, out_features=212)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, label=None):
        outputs = self.bert(input_ids, attention_mask, token_type_ids)
        pooler_output = outputs['pooler_output']
        logits = self.linear(pooler_output)

        if label is not None:
            loss = log_sum_exp(logits, label).mean()
            return loss, logits

        return logits

