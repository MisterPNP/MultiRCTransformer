import glob
import json
import torch

from data_load import encode_tokens, bert_tokenizer, tokenizer


class QADataset(torch.utils.data.Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.embeddings[idx],
            'labels': self.labels[idx],
        }

    def __len__(self):
        return len(self.labels)


def tokenize(string):
    tokens = list(map(str, tokenizer.tokenize(string)))
    embedding, _ = encode_tokens(tokens)
    return embedding


def load_dataset_from_json(path, x=0, y=0):
    start_emb = [torch.tensor(bert_tokenizer.cls_token_id)]
    end_emb = [torch.tensor(bert_tokenizer.sep_token_id)]

    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)['data']

    embeddings = []
    labels = []
    for i, passage in enumerate(data[x:]):
        # "<b>Sent #</b>text...<br>" for each sentence
        text = passage['paragraph']['text'][:-4]
        sentences = [sent.split('</b>')[1] for sent in text.split('<br>')]
        text_emb = list(map(tokenize, sentences))

        for j, question in enumerate(passage['paragraph']['questions'][y:]):
            q_emb, _ = tokenize(question['question'])
            print(x + i, y + j)

            for k, answer in enumerate(question['answers']):
                ans_emb = tokenize(answer['text'])

                all_emb = start_emb + q_emb + ans_emb + text_emb + end_emb
                padding = 512 - len(all_emb)
                all_emb = (all_emb + [torch.tensor(0)] * padding) if padding > 0 else all_emb[:512]
                embedding = torch.stack(all_emb)

                embeddings.append(embedding)
                labels.append(answer['isAnswer'])

    return QADataset(torch.stack(embeddings), torch.LongTensor(labels))


def load_dataset_from_cache(path):
    embeddings = []
    labels = []
    for q_labels in glob.glob(path + "labels*.pt"):
        q_embeddings = "embeddings".join(q_labels.rsplit("labels", maxsplit=1))
        embeddings.append(torch.load(q_embeddings))
        labels.append(torch.load(q_labels))

    return QADataset(torch.concat(embeddings), torch.concat(labels))
