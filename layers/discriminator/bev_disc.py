import torch.nn as nn
import torch
import torch.nn.functional as F

class Disc_bev_source(nn.Module):
    
    def __init__(self):
        super(Disc_bev_source, self).__init__()
        # [4, 80, 128, 128]
        self.conv1 = nn.Conv2d(80, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,16, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
        self.linear1 = torch.nn.Linear(2304,256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,2)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.adv_loss = torch.nn.NLLLoss()

    def forward(self, input):
        x = F.relu(F.max_pool2d(self.conv1(input),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)),2))
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu(self.bn(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x
            
    def set_optimizer(self,param, lr):
        self.optimizer = torch.optim.Adam(param, lr)

    def train_source(self, img, label, param, lr):
        self.set_optimizer(param, lr)
        optimizer = self.optimizer
        optimizer.zero_grad()
        out = self.forward(img)
        loss = self.adv_loss(out, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        return out, loss.item()

class Disc_bev_target(nn.Module):

    def __init__(self):
        super(Disc_bev_target, self).__init__()
        # [4, 80, 128, 128]
        self.conv1 = nn.Conv2d(80, 64, kernel_size=5)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,16, kernel_size=5)
        self.conv3_drop = nn.Dropout2d()
        self.pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.flat = torch.nn.Flatten(1) # ([4, 6*512*16*44])
        self.linear1 = torch.nn.Linear(2304,256)
        self.bn = torch.nn.BatchNorm1d(256)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256,2)
        self.softmax = torch.nn.LogSoftmax(dim=1)
        self.adv_loss = torch.nn.NLLLoss()

    def forward(self, input):
        x = F.relu(F.max_pool2d(self.conv1(input),2))
        x = F.relu(F.max_pool2d(self.conv2(x),2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)),2))
        x = self.flat(x)
        x = self.linear1(x)
        x = self.relu(self.bn(x))
        x = self.linear2(x)
        x = self.softmax(x)
        return x
            
    def set_optimizer(self,param, lr):
        self.optimizer = torch.optim.Adam(param, lr)

    def train_target(self, img, label, param, lr):
        self.set_optimizer(param, lr)
        optimizer = self.optimizer
        optimizer.zero_grad()
        out = self.forward(img)
        loss = self.adv_loss(out, label)
        loss.backward(retain_graph=True)
        optimizer.step()

        return out, loss.item()