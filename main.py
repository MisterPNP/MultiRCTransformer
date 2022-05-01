import os
import torch

import data_load
import data_load_baseline
import data_load_composite
import model_baseline
import model_composite
import setup


use_cuda = True

# Uncomment this line on the first run.
# setup.setup()

# Uncomment these lines on the first run to do preprocessing for the composite
# model (preprocessing also caches the data used by the baseline model). If you
# just want to train the baseline model, you can skip these lines and change
# the calls below from `load_dataset_from_cache` to `load_dataset_from_json`
# data_load.create_cache_from_json(setup.multirc_train_path, setup.preprocessed_train_path, use_cuda=use_cuda)
# data_load.create_cache_from_json(setup.multirc_dev_path, setup.preprocessed_dev_path, use_cuda=use_cuda)


# train and evaluate baseline model
def train_baseline_model():
    train_dataset = data_load_baseline.load_dataset_from_cache(setup.preprocessed_train_path)
    dev_dataset = data_load_baseline.load_dataset_from_cache(setup.preprocessed_dev_path)
    # train_dataset = data_load_baseline.load_dataset_from_json(setup.multirc_train_path)
    # dev_dataset = data_load_baseline.load_dataset_from_json(setup.multirc_dev_path)
    baseline_model = model_baseline.get_model(use_cuda=use_cuda)
    baseline_trainer = model_baseline.get_trainer(baseline_model, train_dataset, dev_dataset, use_cuda=use_cuda)
    os.environ["WANDB_DISABLED"] = "true"
    baseline_trainer.train()


# train and evaluate composite model
def train_composite_model():
    train_dataset = data_load_composite.load_dataset_from_cache(setup.preprocessed_train_path)
    dev_dataset = data_load_composite.load_dataset_from_cache(setup.preprocessed_dev_path)
    classifier = model_composite.FNN(2 * 768, 768)
    if use_cuda:
        classifier = classifier.to('cuda')
    train_loader = torch.utils.data.DataLoader(train_dataset)
    dev_loader = torch.utils.data.DataLoader(dev_dataset)
    bin_cross_ent = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01)
    num_epochs = 1
    model_composite.train(classifier, train_loader, dev_loader, bin_cross_ent, optimizer, num_epochs, use_cuda=use_cuda)


# train_baseline_model()
train_composite_model()
