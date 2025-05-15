import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np



class NTMReadHead(nn.Module):
    def __init__(self, memory, controller_size):
        super(NTMReadHead, self).__init__()

        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

        self.read_lengths = [self.M, 1, 1]
        self.fc_read = nn.Linear(controller_size, sum(self.read_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return Variable(torch.zeros(batch_size, 1, self.N))

    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc_read.weight, gain=1.4)
        nn.init.normal(self.fc_read.bias, std=0.01)

    def is_read_head(self):
        return True


    def _address_memory(self, k, β, g, n, w_prev, access):
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        gamma = 0.95


        w_r = self.memory.address(k, β, g, n, gamma, w_prev, access)
        return w_r

    def forward(self, embeddings, w_prev, n):
        
        o = self.fc_read(embeddings)
        l = np.cumsum([0] + self.read_lengths)
        results = []
        for s, e in zip(l[:-1], l[1:]):
            results += [o[:, s:e]]
        k, β, g = results

        w_r = self._address_memory(k, β, g, n, w_prev, 1)
        r = self.memory.read(w_r)

        return r, w_r
    
    def save_weights(self,location):
        torch.save(self.fc_read.state_dict(), location+"read_head")
        
    def load_weights(self,location):
        self.fc_read.load_state_dict(torch.load(location+"read_head", weights_only=True))
        
        
        
class NTMWriteHead(nn.Module):
    def __init__(self, memory, controller_size):
        super(NTMWriteHead, self).__init__()
                
        self.memory = memory
        self.N, self.M = memory.size()
        self.controller_size = controller_size

        self.write_lengths = [self.M, 1, 1]
        self.fc_write = nn.Linear(controller_size, sum(self.write_lengths))
        self.reset_parameters()

    def create_new_state(self, batch_size):
        return Variable(torch.zeros(batch_size, 3, self.N))

    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc_write.weight, gain=1.4)
        nn.init.normal(self.fc_write.bias, std=0.01)

    def is_read_head(self):
        return False

    def forward(self, embeddings, w_prev, n):

        o = self.fc_write(embeddings)
        
        l = np.cumsum([0] + self.write_lengths)
        results = []
        for s, e in zip(l[:-1], l[1:]):
            results += [o[:, s:e]]
        k, β, g = results

        w_u, w_r, w_w, w_lu = self._address_memory(k, β, g, n, w_prev, 0)

        self.memory.lrua_write(w_w, k)

        w = torch.cat((w_u, w_r, w_lu), dim=1).view(w_u.size()[0], 3, w_u.size()[1])

        return w
    
    def save_weights(self,location):
        torch.save(self.fc_write.state_dict(), location+"write_head")
        
    def load_weights(self,location):
        self.fc_write.load_state_dict(torch.load(location+"write_head", weights_only=True))
        
        
    def _address_memory(self, k, β, g, n, w_prev, access):
        k = k.clone()
        β = F.softplus(β)
        g = F.sigmoid(g)
        gamma = 0.95


        w_u, w_r, w_w, w_lu = self.memory.address(k, β, g, n, gamma, w_prev, access)
        return w_u, w_r, w_w, w_lu