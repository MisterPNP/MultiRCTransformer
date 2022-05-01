from allennlp import predictors
from allennlp.data import tokenizers
import json
import pickle
import torch
import transformers

from setup import coref_path, dep_path


tokenizer = tokenizers.spacy_tokenizer.SpacyTokenizer()
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')


def encode_tokens(tokens):
    bert_tokens = []
    bert_indices = []
    for token in tokens:
        bert_indices.append(len(bert_tokens))
        bert_tokens += bert_tokenizer.encode(token, return_tensors="pt", add_special_tokens=False).squeeze(axis=0)
    return bert_tokens, bert_indices


def create_cache_from_json(path, dest, x=0, y=0, use_cuda=True):
    encoder = transformers.BertModel.from_pretrained("bert-base-cased")
    if use_cuda:
        encoder = encoder.to('cuda')
    device = 0 if use_cuda else -1
    coref = predictors.Predictor.from_path(coref_path, cuda_device=device)
    dep = predictors.Predictor.from_path(dep_path, cuda_device=device)

    start_emb = [torch.tensor(bert_tokenizer.cls_token_id)]
    end_emb = [torch.tensor(bert_tokenizer.sep_token_id)]

    with open(path, 'r', encoding='utf-8') as fp:
        data = json.load(fp)['data']

    for i, passage in enumerate(data[x:]):
        # "<b>Sent #</b>text...<br>" for each sentence
        text = passage['paragraph']['text'][:-4]
        sentences = [sent.split('</b>')[1] for sent in text.split('<br>')]

        text_tokens = []
        text_dep = []
        text_emb = []
        text_idx = []
        sentence_offsets = []
        for sentence in sentences:
            sentence_dep = dep.predict(sentence)
            sentence_tokens = sentence_dep['words']
            sentence_emb, sentence_idx = encode_tokens(sentence_tokens)
            sentence_offsets.append(len(text_tokens))
            text_tokens += sentence_tokens
            text_dep += list(map(add(len(text_dep)), sentence_dep['predicted_heads']))
            text_idx += [index + len(text_emb) for index in sentence_idx]
            text_emb += sentence_emb
        with open(f'{dest}text_{x + i:03d}.pkl', 'wb') as f:
            pickle.dump({'tokens': text_tokens, 'offsets': sentence_offsets,
                         'dep': text_dep, 'emb': text_emb, 'idx': text_idx}, f)

        for j, question in enumerate(passage['paragraph']['questions'][y:]):
            q_tokens = list(map(str, tokenizer.tokenize(question['question'])))
            q_emb, q_idx = encode_tokens(q_tokens)
            with open(f'{dest}question_{x + i:03d}_{y + j:02d}.pkl', 'wb') as f:
                pickle.dump({'tokens': q_tokens, 'emb': q_emb, 'idx': q_idx}, f)

            embeddings = []
            queries = []
            labels = []
            print(x + i, y + j)
            for k, answer in enumerate(question['answers']):
                ans_tokens = list(map(str, tokenizer.tokenize(answer['text'])))
                ans_emb, ans_idx = encode_tokens(ans_tokens)
                with open(f'{dest}answer_{x + i:03d}_{y + j:02d}_{k:02d}.pkl', 'wb') as f:
                    pickle.dump({'tokens': ans_tokens, 'emb': ans_emb, 'idx': ans_idx}, f)

                all_emb = start_emb + q_emb + ans_emb + text_emb + end_emb
                padding = 512 - len(all_emb)
                all_emb = (all_emb + [torch.tensor(0)] * padding) if padding > 0 else all_emb[:512]
                embedding = torch.stack(all_emb)
                if use_cuda:
                    embedding = embedding.to('cuda')
                encoding = encoder(embedding.unsqueeze(0))['last_hidden_state'].to('cpu').squeeze(0)
                # torch.save(encoding, f'{dest}encoding_{x + i:03d}_{y + j:02d}_{k:02d}.pt')

                qa_end = 1 + len(q_emb) + len(ans_emb)
                qa_avg = torch.mean(encoding[1:qa_end], dim=0)

                clusters = coref.predict_tokenized(text_tokens + q_tokens + ans_tokens)['clusters']
                with open(f'{dest}clusters_{x + i:03d}_{y + j:02d}_{k:02d}.pkl', 'wb') as f:
                    pickle.dump(clusters, f)
                clusters = filter(cutoff(len(text_tokens)), clusters)
                clusters = map(spans_to_heads(text_dep, encoding[qa_end:], text_idx), clusters)
                heads = [head for cluster in clusters for head in cluster]
                heads_avg = torch.mean(torch.stack(heads if len(heads) > 0 else [torch.zeros(768)]), dim=0)

                embeddings.append(embedding.to('cpu'))
                queries.append(torch.cat([qa_avg, heads_avg]))
                labels.append(answer['isAnswer'])

            torch.save(torch.stack(embeddings), f'{dest}embeddings_{x + i:03d}_{y + j:02d}.pt')
            torch.save(torch.stack(queries), f'{dest}queries_{x + i:03d}_{y + j:02d}.pt')
            torch.save(torch.LongTensor(labels), f'{dest}labels_{x + i:03d}_{y + j:02d}.pt')
            del queries, labels


def add(sentence_start):
    def f(index):
        return index + sentence_start if index > 0 else index
    return f


def cutoff(question_start):
    def f(cluster):
        return cluster[-1][0] >= question_start
    return f


def spans_to_heads(heads, encodings, indices):
    max_len = len(indices)
    encoding_len = len(encodings)

    def f(cluster):
        out = []
        for [begin, end] in cluster:
            if end >= max_len:
                end = max_len - 1
            span = range(begin, end + 1)
            for i in span:
                head = heads[i] - 1
                if (head not in span) and head > -1:
                    encoding_index = indices[head]
                    if encoding_index < encoding_len:
                        out.append(encodings[indices[head]])
                    break
        return out
    return f
