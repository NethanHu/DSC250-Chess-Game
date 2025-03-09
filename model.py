import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib

matplotlib.use("Agg")

"""
The purpose of the board_data class is to encapsulate a NumPy-format game dataset, 
such as (status, policy, value), into a standard PyTorch dataset for model training.
If the input data format needs to be modified, it should be done here.
e.g.: The data format here is exemplified as (batch_size, 22, 8, 8).
"""


# Custom Dataset class to handle X and y pairs
class ChessDataset(Dataset):
    def __init__(self, X, param_dict):
        self.X = X
        self.best_move = param_dict['best_move'] # need processing embedding -> [73 * 8 * 8]
        self.winner = param_dict['winner']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        # Convert best_move and winner to tensors
        best_move_tensor = torch.tensor(self.best_move[idx], dtype=torch.float32)
        winner_tensor = torch.tensor(self.winner[idx], dtype=torch.int8)
        # Return X[idx] and the processed tensors
        return X_tensor, (best_move_tensor, winner_tensor)


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        # The chessboard has an 8x8 grid, and each position can have 73 possible moves 
        # (e.g., movement directions, captures, etc.)
        # self.action_size = 8 * 8 * 73  # Temporarily unused
        # A convolutional layer set up purely to match the current input requirements
        # self.conv0 = nn.Conv2d(13, 22, 3, stride=1, padding=1)
        # self.bn0 = nn.BatchNorm2d(22)
        # Increase the number of channels from the input 22 to 256, extracting 256 local feature maps
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)


    def forward(self, s):
        # print("Size of input to ConvBlock:", s.shape)
        # s = s.reshape(-1, 13, 8, 8)  # batch_size x channels x board_x x board_y
        # s = F.relu(self.bn0(self.conv0(s)))
        s = s.reshape(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s  # (batch_size, 256, 8, 8)


class ResBlock(nn.Module):
    # Dim of input and output are all 256
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        # Input dimension (batch_size, inplanes, H, W)
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out  # Output dimension (batch_size, planes, H, W)


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # Number of input channels: 256, Number of output channels: 1，(batch_size, 1, 8, 8)
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1)  # Convert to a scalar

        # Number of input channels: 256, Number of output channels: 128
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73)  # Map the input size from (8x8*128) to the action space size 8x8*73 = 4672


    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        # v = v.reshape(-1, 8 * 8)  # batch_size X channel X height X width
        v = self.flatten(v)
        v = F.relu(self.fc1(v))
        v = F.relu(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        # p = p.reshape(-1, 8 * 8 * 128)
        p = self.flatten(p)
        p = self.fc(p)
        p = self.logSoftmax(p).exp()
        return {'p': p, 'v': v}  # Obtain policy distribution (p) and position value (v)
        # p: (batch_size, 8*8*73), represents the probability distribution of all possible moves for each sample
        # v: Represents the evaluated positional advantage of each sample, with values in the range [-1, 1]


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19): # from 19 -> 9 -> 4
            setattr(self, "res_%i" % block, ResBlock())
        self.outblock = OutBlock()
        # self.outblock = SimpleOutBlock()

    def forward(self, s):
        s = self.conv(s)
        # After ConvBlock being processed (batch_size, 22, 8, 8) -> (batch_size, 256, 8, 8)
        for block in range(4): # from 19 -> 9 -> 4
            # (batch_size, 256, 8, 8) -> (batch_size, 256, 8, 8)
            s = getattr(self, "res_%i" % block)(s)
        res_dict = self.outblock(s)
        return res_dict


class SimpleOutBlock(nn.Module):
    def __init__(self):
        super(SimpleOutBlock, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Linear(256 * 8 * 8, 2048)  # Flattened board (256 channels x 8 x 8 positions)
        self.relu = nn.ReLU()
        self.output = nn.Linear(2048, 1)  # Output layer with 1 unit

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.hidden(x))
        x = self.output(x)
        return x
