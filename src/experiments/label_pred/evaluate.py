# -*- coding: utf-8 -*-
# @Time : 3/29/21 2:54 PM
# @Author : leoyao
# -*- coding: utf-8 -*-
# @Time : 3/28/21 4:13 PM
# @Author : leoyao
import torch
from torch.utils.data import DataLoader
from dataset import CaseData
from model import CaseClassification
from transformers import AutoTokenizer


if __name__ == "__main__":
    # check the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # prepare test data
    test_data = CaseData('./data/level2/test.json')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)

    # load the model and tokenizer
    model = CaseClassification().to(device)
    # model.load_state_dict(torch.load('model_loss_changed.pth'))
    model.load_state_dict(torch.load('./saved/model.pth'))
    tokenizer = AutoTokenizer.from_pretrained('schen/longformer-chinese-base-4096')

    # start evaluating process
    model.eval()
    tp = 0
    tp_plus_fp = 0
    tp_plus_fn = 0
    prediction_result = []
    for i, data in enumerate(test_dataloader):
        print('evaluating ', i, 'th batch')
        content, label = data

        # tokenize the input text
        inputs = tokenizer(list(content), max_length=1300, padding=True, truncation=True, return_tensors='pt')

        # move all data to cuda
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

            prediction = (logits > 0.5).reshape(-1)

            prediction = prediction[80:]

            single_result = []
            for j, pred in enumerate(prediction):
                if pred:
                    single_result.append(j)

            prediction_result.append(single_result)

            target = label.reshape(-1).bool()
            target = target[80:]

            tp += (prediction & target).tolist().count(True)
            tp_plus_fp += prediction.tolist().count(True)
            tp_plus_fn += target.tolist().count(True)

    precision = tp / tp_plus_fp if tp_plus_fp != 0 else 0
    recall = tp / tp_plus_fn if tp_plus_fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall)
    print('precision: %3f, recall:%3f, f1:%3f' % (precision, recall, f1))
