import torch.nn as nn
import torch.nn.functional as F

class CNNLstm(nn.Module):
    """CNN LSTM Model
    
    Arguments:
    __________
    opt_class: int
        Number of Genre Classes to be predicted
        
    """
    def __init__(self, opt_class):
        super().__init__()
        self.conv11 = nn.Conv2d(1, 64, 3, padding = 1)
        self.conv12 = nn.Conv2d(64, 64, 3, padding = 1)
        
        self.conv21 = nn.Conv2d(64, 128, 5, padding = 1)
        self.conv22 = nn.Conv2d(128, 128, 5, padding = 1)
        
        self.conv31 = nn.Conv2d(128, 256, 5, padding = 1)
        self.conv32 = nn.Conv2d(256, 512, 5, padding = 1)
        self.conv33 = nn.Conv2d(512, 1024, 5, padding = 1)
        # self.conv34 = nn.Conv2d(1024, 512, 5, padding = 1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(512)
        self.bn4 = nn.BatchNorm2d(1024)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.pool1 = nn.MaxPool2d(3, 3)
        # self.pool2 = nn.MaxPool2d(5, 5)
        
        self.map = nn.Linear(8192, 512)
        
        self.lstm1 = nn.LSTM(512, 256, num_layers=2, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(512, 256, num_layers=4, batch_first=True, bidirectional=True)
        self.lstm3 = nn.LSTM(512, 256, num_layers=8, batch_first=True, bidirectional=True)
        self.lstm4 = nn.LSTM(512, 256, num_layers=16, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(2560, 1024)
        self.fc2 = nn.Linear(1024, opt_class)
    
    def forward(self, input):
        x = F.relu(self.conv11(input))
        x = F.relu(self.conv12(x))
        x = self.bn1(x)
        x = self.pool(x)
        
        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.bn2(x)
        x = self.pool(x)
        
        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))        
        x = F.relu(self.conv33(x))
        # x = F.relu(self.conv34(x))
        x = self.bn4(x)
        x = self.pool1(x)
        
        batch, channel, height, width = x.size()
        x = x.view(batch, channel*height, width)
        x = x.permute(0, 2, 1)
        seq = self.map(x)
        # print(seq)
        hidden_11, _ = self.lstm1(seq)
        hidden_12, _ = self.lstm2(hidden_11)
        # hidden_13, _ = self.lstm3(hidden_12)
        hidden_21, _ = self.lstm1(seq)
        hidden_22, _ = self.lstm2(hidden_21)
        # hidden_23, _ = self.lstm2(hidden_22)
        hidden = hidden_12 + hidden_22
        
        # print(hidden.size(), hidden_1.size(), hidden_2.size())
        s = hidden.size()
        hidden = hidden.reshape(s[0], s[1]*s[2])
        
        x = F.relu(self.fc1(hidden))
        x =  F.dropout(x, 0.5)
        x = self.fc2(x)
        return x
    