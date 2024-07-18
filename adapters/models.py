import torch.nn as nn
import torch.nn.functional as F

class LinearLayer(nn.Module):
    def __init__(self, d):
        super(LinearLayer, self).__init__()
        self.l1 = nn.Linear(d, d)
    
    def forward(self, X):
        return self.l1(X)
    

#Or try reduced version
class ReducedLinearLayer(nn.Module):
    def __init__(self, d, d_hidden):
        super(ReducedLinearLayer, self).__init__()
        self.l1 = nn.Linear(d, d_hidden)
        self.l2 = nn.Linear(d_hidden, d)
    
    def forward(self, X):
        return self.l2(F.relu(self.l1(X)))
    