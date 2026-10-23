import torch
import torch.nn as nn
import torch.nn.functional as F


class autoencoderMLP4Layer(nn.Module):

    def __init__(self, N_input=784, N_bottleneck=32, N_output=784):
        super(autoencoderMLP4Layer, self).__init__()
        N2 = N_input // 2
        # ---------------------------------
        # Your code here ...
        self.fc1 = nn.Linear(N_input, N2)          # 784 -> 392
        self.fc2 = nn.Linear(N2, N_bottleneck)      # 392 -> B
        self.fc3 = nn.Linear(N_bottleneck, N2)      # B -> 392
        self.fc4 = nn.Linear(N2, N_output)          # 392 -> 784
        self.type = 'MLP4'
        self.inputShape=(1,28*28)

        # ---------------------------------

    def encode(self, X):
        # ---------------------------------
        # Your code here ...
        X= F.relu(self.fc1(X))#relu for layer 1  
        X= F.relu(self.fc2(X))#relu for bottle neck
        return X
        # ---------------------------------

    def decode(self, X):
        # ---------------------------------
        # Your code here ...
        X= F.relu(self.fc3(X))
        X = torch.sigmoid(self.fc4(X))
        return X    
        # ---------------------------------

    def forward(self, X):
        return self.decode(self.encode(X))
