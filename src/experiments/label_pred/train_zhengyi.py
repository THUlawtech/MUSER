import torch
from torch.utils.data import DataLoader
from dataset import SentenceData_ZY
from model_law import SentenceClassification_ZY
from transformers import AutoTokenizer, AdamW


def train_loop(dataloader, model, optimizer):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        sentence, label = data

        # tokenize the input text
        inputs = tokenizer(list(sentence), padding='longest', truncation='longest_first', return_tensors='pt')

        # move all data to cuda
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        label = label.to(device)

        global_attention_mask = torch.zeros_like(input_ids)
        global_attention_mask = torch.index_fill(global_attention_mask, dim=1, index=torch.tensor(1).to(device), value=1)

        # forward and backward propagations
        loss, logits = model(input_ids, attention_mask, token_type_ids, global_attention_mask, label)
        # loss, logits = model(input_ids, attention_mask, token_type_ids, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('epoch%d, step%6d, loss: %.3f' % (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0


def test_loop(dataloader, model):
    model.eval()
    predictions = []
    targets = []
    for i, data in enumerate(dataloader):
        sentence, label = data

        # tokenize the input text
        inputs = tokenizer(list(sentence), padding='longest', truncation='longest_first', return_tensors='pt')

        # move all data to cuda
        input_ids = inputs['input_ids'].to(device)
        token_type_ids = inputs['token_type_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        # for level in label.keys():
        #     label[level] = label[level].to(device)
        label = label.to(device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask, token_type_ids)

            prediction = logits > 0

            predictions.append(prediction)
            targets.append(label)

    predictions = torch.cat(predictions, dim=0)
    targets = torch.cat(targets, dim=0)

    pred = predictions.to(device).bool().view(-1)
    gold = targets.to(device).bool().view(-1)

    tp = (pred & gold).tolist().count(True)
    tp_plus_fp = pred.tolist().count(True)
    tp_plus_fn = gold.tolist().count(True)

    precision = tp / tp_plus_fp if tp_plus_fp != 0 else 0
    recall = tp / tp_plus_fn if tp_plus_fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    print('all: precision: %3f, recall:%3f, f1:%3f' % (precision, recall, f1))


if __name__ == "__main__":
    # check the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    # prepare training  data
    train_data = SentenceData_ZY('./民间借贷6批结果9.2/all_train_sent_data_zy.json')
    train_dataloader = DataLoader(train_data, batch_size=1, shuffle=True)

    test_data = SentenceData_ZY('./民间借贷6批结果9.2/all_test_sent_data_zy.json')
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=False)

    # load the model and tokenizer
    model = SentenceClassification_ZY().to(device)
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")

    # prepare the optimizer and corresponding hyper-parameters
    epochs = 20
    learning_rate = 1e-5
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # start training process
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        train_loop(train_dataloader, model, optimizer)
        test_loop(test_dataloader, model)
        torch.save(model.state_dict(), './saved_zhengyi/model'+str(epoch+1)+'.pth')
