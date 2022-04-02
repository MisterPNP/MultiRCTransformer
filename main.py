import torch
import transformers
import sklearn
import re
import json
import numpy as np

# load data from json files
train_path = "./MultiRC Data/splitv2/train_456-fixedIds.json"
test_path = "./MultiRC Data/splitv2/dev_83-fixedIds.json"
with open(train_path, "r", encoding='utf-8') as fp:
    train_data = json.load(fp)['data']
with open(test_path, "r", encoding='utf-8') as fp:
    test_data = json.load(fp)['data']

pretrained_model_ver = 'bert-base-cased'
tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_ver)


def tokenize(text):
    return tokenizer(text, truncation=True, padding=True, max_length=512)


def get_encodings_and_labels(data):
    texts = []
    labels = []
    for passage in data:
        # "<b>Sent #</b>text...<br>" for each sentence
        text = passage['paragraph']['text']
        text = re.sub('<b>.*?</b>', '', text)
        text = ' '.join(text.split('<br>'))
        for question in passage['paragraph']['questions']:
            for answer in question['answers']:
                texts.append(' '.join([text, question['question'], answer['text']]))
                labels.append(answer['isAnswer'])
    return tokenize(texts), np.array(labels)

class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)


# convert our tokenized data into a torch Dataset
train_dataset = QADataset(*get_encodings_and_labels(train_data))
test_dataset = QADataset(*get_encodings_and_labels(test_data))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = sklearn.metrics.accuracy_score(labels, preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, preds).ravel()
    return {
        'accuracy': acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp
    }


model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_ver, num_labels=2)
# model = model.to('cuda')
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=transformers.TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=20,  # batch size per device during training
        per_device_eval_batch_size=20,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        logging_dir='./logs',  # directory for storing logs
        load_best_model_at_end=True,  # load the best model when finished training (default metric is loss)
        # but you can specify `metric_for_best_model` argument to change to accuracy or other metric
        logging_steps=200,  # log & save weights each logging_steps
        save_steps=200,
        evaluation_strategy="steps",  # evaluate each `logging_steps`?
    ),
    compute_metrics=compute_metrics,
)
trainer.train()
