import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class NTM(nn.Module):
    def __init__(self, num_outputs, controller, memory, heads):
        super(NTM, self).__init__()
        
        self.controller = controller
        
        self.memory = memory
        
        self.heads = heads

        _, self.M = memory.size()
        _, self.controller_size = controller.size()

        self._set_head()
        self.fc = nn.Linear(self.controller_size + (self.num_read_heads * self.M), num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        
        controller_state = self.controller.create_new_state(batch_size)
        
        heads_state = [head.create_new_state(batch_size) for head in self.heads]

        return init_r, controller_state, heads_state

    def _set_head(self):
        self.num_read_heads = 0
        self.init_r = []
        for head in self.heads:
            if head.is_read_head():
                init_r_bias = Variable(torch.randn(1, self.M) * 0.01)
                self.register_buffer("read{}_bias".format(self.num_read_heads), init_r_bias.data)
                self.init_r += [init_r_bias]
                self.num_read_heads += 1


    def reset_parameters(self):
        nn.init.xavier_uniform(self.fc.weight, gain=1)
        nn.init.normal(self.fc.bias, std=0.01)

    def forward(self, x, prev_state, read_only=False):

        prev_reads, prev_controller_state, prev_heads_states = prev_state

        inp = torch.cat([x] + prev_reads, dim=1)
        controller_outp, controller_state = self.controller(inp, prev_controller_state)

        reads = []
        heads_states = []
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state, self.num_read_heads)
                reads += [r]

            else:
                if not read_only:
                    head_state = head(controller_outp, prev_head_state, self.num_read_heads)
            heads_states += [head_state]

        ntm_out = torch.cat([controller_outp] + reads, dim=1)
        predictions = self.fc(ntm_out)

        state = (reads, controller_state, heads_states)

        return predictions, state
