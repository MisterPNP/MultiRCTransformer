import torch
import transformers
import sklearn
import json


pretrained_model_ver = 'bert-base-cased'
tokenizer = transformers.BertTokenizer.from_pretrained(pretrained_model_ver)


class QADataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[:, idx],
            'labels': self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


def load_dataset(path):
    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)['data']

    encodings = []
    labels = []
    for passage in data:
        # "<b>Sent #</b>text...<br>" for each sentence
        text = passage['paragraph']['text'][:-4]
        text = ' '.join(sent.split('</b>')[1] for sent in text.split('<br>'))
        text_enc = tokenizer.encode(text)
        for question in passage['paragraph']['questions']:
            q_enc = tokenizer.encode(question['question'])
            for answer in question['answers']:
                ans_enc = tokenizer.encode(answer['text'])
                encodings.append(torch.tensor(ans_enc + q_enc + text_enc))
                labels.append(answer['isAnswer'])

    encodings = torch.nn.utils.rnn.pad_sequence(encodings)[:512]
    labels = torch.tensor(labels).long()
    return QADataset(encodings, labels)


train_dataset = load_dataset('./MultiRC Data/splitv2/train_456-fixedIds.json')
test_dataset = load_dataset('./MultiRC Data/splitv2/dev_83-fixedIds.json')


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = sklearn.metrics.accuracy_score(labels, preds)
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(labels, preds).ravel()
    return {'accuracy': acc, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}


model = transformers.AutoModelForSequenceClassification.from_pretrained(pretrained_model_ver, num_labels=2)
model = model.to('cuda')
trainer = transformers.Trainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    args=transformers.TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # total number of training epochs
        per_device_train_batch_size=64,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,  # number of warmup steps for learning rate scheduler
        weight_decay=0.01,  # strength of weight decay
        # load_best_model_at_end=True,
        # -> When set to True, the parameter save_strategy needs to be the same as eval_strategy,
        #    and in the case it is “steps”, save_steps must be a round multiple of eval_steps.
        # metric_for_best_model='eval_tp',  # defualts to 'loss'
        logging_strategy='no',
        # logging_steps=200,  # log each logging_steps
        # logging_dir='./logs',  # directory for storing logs
        save_strategy='no',
        # save_steps=200,  # save weights each save_steps
        evaluation_strategy='steps',
        eval_steps=200,  # evaluate each eval_steps
        # log_level='debug',
        # skip_memory_metrics=False,
        gradient_checkpointing=True,  # saves memory, might be slower
        # no_cuda=True,  # force trainer to use CPU
        # fp16=True,  # use half-precision floats
    ),
    compute_metrics=compute_metrics,
)
trainer.train()
