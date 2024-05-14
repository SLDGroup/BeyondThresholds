import torch
import torch.nn as nn
import torch.nn.functional as F
from ptflops import get_model_complexity_info

class DSADSNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, 5, padding=1)
        self.conv2 = nn.Conv1d(16, 16, 3, 3)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.conv4 = nn.Conv1d(16, 16, 3)
        self.conv5 = nn.Conv1d(16, 32, 12)
        # self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,19)
    

    def forward(self, x):
        # print(x.shape)
        # (3 x 50) --> (32 x 48)
        x = F.relu(self.conv1(x))

        # (32 x 48) --> (32 x 16)
        # print(x.shape)
        x = F.relu(self.conv2(x))

        # (32 x 16) --> (32 x 14)
        # print(x.shape)
        x = F.relu(self.conv3(x))

        # (32 x 14) --> (32 x 12)
        # print(x.shape)
        x = F.relu(self.conv4(x))

        # (32 x 12) --> (64 x 1)
        # print(x.shape)
        x = F.relu(self.conv5(x))

        # (64 x 1) --> (64) --> (32)
        x = x.view(x.shape[0],-1)
        # x = F.relu(self.fc1(x))

        # (32) --> (19)
        # print(x.shape)
        x = self.fc2(x)

        return x


class RWHARNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, 5)
        self.conv2 = nn.Conv1d(16, 16, 3, 3)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.conv4 = nn.Conv1d(16, 16, 3, 3)
        self.conv5 = nn.Conv1d(16, 32, 10)
        # self.fc1 = nn.Linear(64,32)
        self.fc2 = nn.Linear(32,8)
    

    def forward(self, x):
        # print(x.shape)
        # (3 x 100) --> (16 x 96)
        x = F.relu(self.conv1(x))

        # (16 x 96) --> (32 x 32)
        # print(x.shape)
        x = F.relu(self.conv2(x))

        # (32 x 32) --> (32 x 30)
        # print(x.shape)
        x = F.relu(self.conv3(x))

        # (32 x 30) --> (32 x 10)
        # print(x.shape)
        x = F.relu(self.conv4(x))

        # (32 x 10) --> (64 x 1)
        # print(x.shape)
        x = F.relu(self.conv5(x))

        # (64 x 1) --> (64) --> (32)
        x = x.view(x.shape[0],-1)
        # x = F.relu(self.fc1(x))

        # (32) --> (8)
        # print(x.shape)
        x = self.fc2(x)

        return x
    

class OppNet(nn.Module):
    def __init__(self,in_channels):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, 16, 5, padding=2)
        self.conv2 = nn.Conv1d(16, 16, 3, 3)
        self.conv3 = nn.Conv1d(16, 16, 3)
        self.conv4 = nn.Conv1d(16, 16, 3)
        self.conv5 = nn.Conv1d(16, 24, 16)
        # self.fc1 = nn.Linear(48,32)
        self.fc2 = nn.Linear(24,5)
    

    def forward(self, x):
        # print(x.shape)
        # (3 x 60) --> (32 x 60)
        x = F.relu(self.conv1(x))

        # (32 x 60) --> (32 x 20)
        # print(x.shape)
        x = F.relu(self.conv2(x))

        # (32 x 20) --> (32 x 18)
        # print(x.shape)
        x = F.relu(self.conv3(x))

        # (32 x 18) --> (32 x 16)
        # print(x.shape)
        x = F.relu(self.conv4(x))

        # (32 x 16) --> (48 x 1)
        # print(x.shape)
        x = F.relu(self.conv5(x))

        # (48 x 1) --> (48) --> (32)
        x = x.view(x.shape[0],-1)
        # x = F.relu(self.fc1(x))

        # (32) --> (5)
        # print(x.shape)
        x = self.fc2(x)

        return x


def print_model_size(model,in_dim):
    macs, params = get_model_complexity_info(model, in_dim, as_strings=True,
                                           print_per_layer_stat=False, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

if __name__ == '__main__':

    m = DSADSNet(3)
    x = torch.rand(32,3,50)
    print(m(x).shape)
    print_model_size(m,tuple(x.shape[1:]))

    m = RWHARNet(3)
    x = torch.rand(32,3,100)
    print(m(x).shape)
    print_model_size(m,tuple(x.shape[1:]))

    m = OppNet(3)
    x = torch.rand(32,3,60)
    print(m(x).shape)
    print_model_size(m,tuple(x.shape[1:]))