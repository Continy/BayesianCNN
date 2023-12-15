import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34


class BayesianCNN(nn.Module):

    def __init__(self, n_classes=10, hidden_dim=256):
        super(BayesianCNN, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.rescov = resnet34(pretrained=True)
        self.resmodules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*self.resmodules)
        self.rescov = nn.Sequential(*list(self.rescov.children())[:-1])
        self.log_var = nn.Linear(hidden_dim, hidden_dim)
        self.log_var_bn = nn.BatchNorm1d(hidden_dim)
        self.log_var.weight.data.fill_(0)
        self.log_var.bias.data.fill_(0)
        self.mean = nn.Linear(hidden_dim, hidden_dim)
        self.mean_bn = nn.BatchNorm1d(hidden_dim)
        self.mean.weight.data.fill_(0)
        self.mean.bias.data.fill_(0)
        self.fc1 = nn.Linear(hidden_dim, n_classes)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        mean = self.resnet(x)
        cov = self.rescov(x)
        mean = mean.view(x.size(0), -1)
        cov = cov.view(cov.size(0), -1)
        log_var = self.log_var(cov)
        log_var = self.log_var_bn(log_var)
        mean = self.mean(mean)
        mean = self.mean_bn(mean)
        mean = F.softmax(self.fc1(mean), dim=1)
        var = F.softplus(self.fc2(log_var), beta=1, threshold=20)
        return mean, var


class CNN(nn.Module):

    def __init__(self, n_classes=10, hidden_dim=256):
        super(CNN, self).__init__()
        self.resnet = resnet34(pretrained=True)
        self.resmodules = list(self.resnet.children())[:-1]
        self.resnet = nn.Sequential(*self.resmodules)
        self.log_var = nn.Linear(hidden_dim, hidden_dim)
        self.log_var_bn = nn.BatchNorm1d(hidden_dim)
        self.log_var.weight.data.fill_(0)
        self.log_var.bias.data.fill_(0)
        self.mean = nn.Linear(hidden_dim, hidden_dim)
        self.mean_bn = nn.BatchNorm1d(hidden_dim)
        self.mean.weight.data.fill_(0)
        self.mean.bias.data.fill_(0)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        mean = self.mean(x)
        mean = self.mean_bn(mean)

        x = mean
        out = self.fc(x)
        return out
