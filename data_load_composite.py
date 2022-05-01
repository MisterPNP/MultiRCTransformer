import glob
import torch


class QADataset(torch.utils.data.Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels

    def __getitem__(self, idx):
        return self.queries[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)


def load_dataset_from_cache(path):
    queries = []
    labels = []
    for q_labels in glob.glob(path + "labels*.pt"):
        q_queries = "queries".join(q_labels.rsplit("labels", maxsplit=1))
        queries.append(torch.load(q_queries))
        labels.append(torch.load(q_labels))

    return QADataset(torch.concat(queries), torch.concat(labels).float().unsqueeze(-1))
