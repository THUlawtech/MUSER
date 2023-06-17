import torch.nn as nn
from transformers import AutoModel
from LogSumExpLoss import log_sum_exp


class SentenceClassification(nn.Module):
    def __init__(self):
        super(SentenceClassification, self).__init__()
        self.lawformer = AutoModel.from_pretrained('thunlp/Lawformer')
        self.linear = nn.Linear(in_features=768, out_features=212)               

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                global_attention_mask=None, label=None):
        outputs = self.lawformer(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 global_attention_mask=global_attention_mask)
        pooler_output = outputs['pooler_output']
        logits = self.linear(pooler_output)

        if label is not None:
            loss = log_sum_exp(logits, label).mean()
            return loss, logits

        return logits

class SentenceClassification_ZY(nn.Module):
    def __init__(self):
        super(SentenceClassification_ZY, self).__init__()
        self.lawformer = AutoModel.from_pretrained('thunlp/Lawformer')
        self.linear = nn.Linear(in_features=768, out_features=295)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                global_attention_mask=None, label=None):
        outputs = self.lawformer(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 token_type_ids=token_type_ids,
                                 global_attention_mask=global_attention_mask)
        pooler_output = outputs['pooler_output']
        logits = self.linear(pooler_output)

        if label is not None:
            loss = log_sum_exp(logits, label).mean()
            return loss, logits

        return logits

