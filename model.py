import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib

matplotlib.use("Agg")

"""
board_data 类的作用是将一个 NumPy 格式的对局数据集
    如 (status, policy, value) 封装成 PyTorch 标准数据集，用于训练模型。
** 如果需要修改传入的数据格式，应该在这里进行修改
e.g.: 这里的数据格式以 (batch_size, 22, 8, 8) 举例
"""


class BoardData(Dataset):
    def __init__(self, dataset):  # dataset = np.array of (s, p, v)
        self.X = dataset[:, 0]
        self.y_p, self.y_v = dataset[:, 1], dataset[:, 2]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Convert data to PyTorch tensors and ensure float type for the input
        board = torch.tensor(self.X[idx].transpose(2, 0, 1), dtype=torch.float)
        policy = torch.tensor(self.y_p[idx], dtype=torch.float)
        value = torch.tensor(self.y_v[idx], dtype=torch.float)
        return board, policy, value


class ConvBlock(nn.Module):
    def __init__(self):
        super(ConvBlock, self).__init__()
        # 棋盘有 8x8 的格子，每个位置可能有 73 种可能的动作（如移动方向、吃子等）
        self.action_size = 8 * 8 * 73 # 暂时先不使用
        self.conv0 = nn.Conv2d(13, 22, 3, stride=1, padding=1) # 设置这个纯粹是为了满足当前输入的需要
        self.bn0 = nn.BatchNorm2d(22)
        # 通道数从输入的 22 增加到 256，提取到 256 个局部特征映射
        self.conv1 = nn.Conv2d(22, 256, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

    def forward(self, s):
        print(s.shape)
        # s = s.reshape(-1, 13, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn0(self.conv0(s)))
        # s = s.reshape(-1, 22, 8, 8)  # batch_size x channels x board_x x board_y
        s = F.relu(self.bn1(self.conv1(s)))
        return s  # (batch_size, 256, 8, 8)


class ResBlock(nn.Module):
    # 输入输出都是 256 个通道
    def __init__(self, inplanes=256, planes=256, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        # 输入维度大小 (batch_size, inplanes, H, W)
        residual = x
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = F.relu(out)
        return out  # 输出维度大小 (batch_size, planes, H, W)


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        # 输入通道数为 256，输出通道数为 1，(batch_size, 1, 8, 8)
        self.conv = nn.Conv2d(256, 1, kernel_size=1)  # value head
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(8 * 8, 64)
        self.fc2 = nn.Linear(64, 1) # 转化为一个标量

        # 输入通道数为 256，输出通道数为 128
        self.conv1 = nn.Conv2d(256, 128, kernel_size=1)  # policy head
        self.bn1 = nn.BatchNorm2d(128)
        self.logSoftmax = nn.LogSoftmax(dim=1)
        self.fc = nn.Linear(8 * 8 * 128, 8 * 8 * 73) # 将输入大小从 (8x8*128) 映射到动作空间大小 8x8*73=4672


    def forward(self, s):
        v = F.relu(self.bn(self.conv(s)))  # value head
        # v = v.reshape(-1, 8 * 8)  # batch_size X channel X height X width
        v = F.relu(self.fc1(v))
        v = F.tanh(self.fc2(v))

        p = F.relu(self.bn1(self.conv1(s)))  # policy head
        # p = p.reshape(-1, 8 * 8 * 128)
        p = self.fc(p)
        p = self.logSoftmax(p).exp()
        return p, v  # 获取 策略分布(p) 和 局面价值(v)
        # p: (batch_size, 8*8*73)，表示每个样本的所有动作的可能性概率分布
        # v: 表示每个样本经过评估后的局面优劣值，取值 [-1, 1]


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = ConvBlock()
        for block in range(19):
            setattr(self, "res_%i" % block,ResBlock())
        # self.outblock = OutBlock() # 我们暂时放弃 MCTS 的双输出
        self.outblock = SimpleOutBlock()

    def forward(self,s):
        # 来自于数据集，经过 BoardData 数据加载器加载后的数据
        # 一开始的形状可能是 (batch_size, 22, 8, 8)
        s = self.conv(s)
        # 经过 ConvBlock 之后 (batch_size, 22, 8, 8) -> (batch_size, 256, 8, 8)
        for block in range(19):
            # 定义 19 层的 ResNet，但是每层都不会添加新的通道数
            # (batch_size, 256, 8, 8) -> (batch_size, 256, 8, 8)
            s = getattr(self, "res_%i" % block)(s)
        s = self.outblock(s)
        return s

# 原来的 OutBlock 需要和 MCTS 相结合使用，会输出两个值
# 为了满足此次作业的要求，我们先只输出一个值，为了满足模型 y-label 的需要
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


