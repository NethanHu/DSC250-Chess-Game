import os
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class AlphaLoss(nn.Module):
    def __init__(self):
        super(AlphaLoss, self).__init__()
        self.alpha = 0.9 # value 所占的比重

    def forward(self, y_value, value, y_policy, policy):
        value_error = (value - y_value) ** 2
        policy_error = torch.sum((-policy * (1e-6 + y_policy.float()).float().log()), 1)
        total_error = (self.alpha * value_error.reshape(-1).float() + (1 - self.alpha) * policy_error).mean()
        return total_error


def train(net, train_loader, epoch_start=0, epoch_stop=20, device="cpu"):
    net.train()
    criterion = AlphaLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200, 300, 400], gamma=0.2)

    losses_per_epoch = []
    for epoch in range(epoch_start, epoch_stop):
        scheduler.step()
        total_loss = 0.0
        losses_per_batch = []
        for i, data in enumerate(train_loader, 0):
            state, policy, value = data
            state, policy, value = state.to(device).float(), policy.float().cuda(), value.cuda().float()
            optimizer.zero_grad()
            policy_pred, value_pred = net(
                state)  # policy_pred = torch.Size([batch, 4672]) value_pred = torch.Size([batch, 1])
            loss = criterion(value_pred[:, 0], value, policy_pred, policy)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches of size = batch_size
                print('Process ID: %d [Epoch: %d, %5d/ %d points] total loss per batch: %.3f' %
                      (os.getpid(), epoch + 1, (i + 1) * 30, len(train_loader), total_loss / 10))
                print("Policy:", policy[0].argmax().item(), policy_pred[0].argmax().item())
                print("Value:", value[0].item(), value_pred[0, 0].item())
                losses_per_batch.append(total_loss / 10)
                total_loss = 0.0
        losses_per_epoch.append(sum(losses_per_batch) / len(losses_per_batch))
        if len(losses_per_epoch) > 100:
            if abs(sum(losses_per_epoch[-4:-1]) / 3 - sum(losses_per_epoch[-16:-13]) / 3) <= 0.01:
                break

    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1, epoch_stop + 1, 1)], losses_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')
    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_%s.png" % datetime.today().strftime("%Y-%m-%d")))
