from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F


# feedforward network: binary classification of questions/relations as correct or incorrect
class FNN(nn.Module):

    # n is input dimension, k dimension of hidden layers
    def __init__(self, n, k):
        super(FNN, self).__init__()
        self.layer1 = nn.Linear(n, k)
        self.layer2 = nn.Linear(k, k)
        self.output_layer = nn.Linear(k, 1)

    # x is vector
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x


def train_epoch(model, train_loader, loss_fn, optimizer, use_cuda=True):
    running_loss = 0.
    last_loss = 0.
    for i, data in enumerate(train_loader):
        inputs, labels = data
        if use_cuda:
            inputs = inputs.to('cuda')
            labels = labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            # tb_x = epoch_index * len(training_loader) + i + 1
            # tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss


def train(model, train_loader, dev_loader, loss_fn, optimizer, num_epochs, use_cuda=True):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))

    best_vloss = 1_000_000.
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch + 1))

        model.train(True)
        avg_loss = train_epoch(model, train_loader, loss_fn, optimizer, use_cuda=use_cuda)
        model.train(False)

        running_vloss = 0.0
        for i, vdata in enumerate(dev_loader):
            vinputs, vlabels = vdata
            if use_cuda:
                vinputs = vinputs.to('cuda')
                vlabels = vlabels.to('cuda')
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        # Log the running loss averaged per batch
        # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                    {'Training': avg_loss, 'Validation': avg_vloss},
        #                    epoch_number + 1)
        # writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch)
            torch.save(model.state_dict(), model_path)
