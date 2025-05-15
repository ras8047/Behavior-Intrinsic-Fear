import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np









class NTMMemory(nn.Module):
    def __init__(self, N, M):

        super(NTMMemory, self).__init__()
        self.N = N
        self.M = M
        self.register_buffer('mem_bias', torch.Tensor(N, M))
        stdev = 1 / (np.sqrt(N + M))
        nn.init.uniform(self.mem_bias, -stdev, stdev)

    def reset(self, batch_size):
        self.batch_size = batch_size
        self.memory = Variable(self.mem_bias.clone().repeat(batch_size, 1, 1))

    def size(self):
        return self.N, self.M

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory).squeeze(1)

    def lrua_write(self, w, k):
        self.prev_mem = self.memory
        self.memory = Variable(torch.Tensor(self.batch_size, self.N, self.M))
        lrua = torch.matmul(w.unsqueeze(-1), k.unsqueeze(1))
        self.memory = self.prev_mem + lrua


    def address(self, k, β, g, n, gamma, w_prev, access):
        w_r = self._similarity(k)

        if access == 1:
            return w_r

        w_u_prev = w_prev[:, 0]
        w_r_prev = w_prev[:, 1]
        w_lu_prev = w_prev[:, 2]

        w_w = self._interpolate(w_r_prev, w_lu_prev, g)

        w_u = gamma*w_u_prev + w_r + w_w

        n_smallest_matrix = np.partition(np.array(w_u.data), n-1)[:, n-1]
        w_lu = Variable(torch.FloatTensor(((np.array(w_u.data).transpose() <= n_smallest_matrix).astype(int)).transpose()))

        erase_vector = Variable(torch.ones(w_lu_prev.size()[:]).type(torch.LongTensor)) - w_lu_prev.type(torch.LongTensor)
        zeroed_memory = self.memory.data.clone()
        for b in range(len(erase_vector)):
            for m in range(len(erase_vector[b])):
                if erase_vector.data[b][m] == 0:
                    zeroed_memory[b][m] = torch.zeros(self.M)

        self.memory = Variable(zeroed_memory)

        return w_u, w_r, w_w, w_lu

    def _similarity(self, k):
        k = k.view(self.batch_size, 1, -1)
        w = F.softmax(F.cosine_similarity(self.memory + 1e-16, k + 1e-16, dim=-1), dim=-1)
        return w

    def _interpolate(self, w_r_prev, w_lu_prev, g):
        return g * w_r_prev + (1 - g) * w_lu_prev


    def save_memory(self,location):
        torch.save(self.memory, location+'memory.pth')
        
        
    def load_memory(self,location):
        self.memory=Variable(torch.load(location+'memory.pth'))

        