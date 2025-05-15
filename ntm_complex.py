from torch import nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F








class NTM(nn.Module):
    def __init__(self, num_outputs, controller, memory, heads):
        super(NTM, self).__init__()
        
        
        self.num_outputs = num_outputs
        
        self.controller = controller
        
        self.memory = memory
        self.heads = heads
        _, self.M = memory.size()
        _, self.controller_size = controller.size()

        self._set_heads()

        self.fc = nn.Linear(self.controller_size + (self.num_read_heads * self.M), num_outputs)
        self.reset_parameters()

    def create_new_state(self, batch_size):
        init_r = [r.clone().repeat(batch_size, 1) for r in self.init_r]
        controller_state = self.controller.create_new_state(batch_size)
        heads_state = [head.create_new_state(batch_size) for head in self.heads]
        return init_r, controller_state, heads_state


    def _set_heads(self):
        
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

    def forward(self, x, prev_state, delimeter,class_vector=None, read_only=False,seq=3):

        prev_reads, prev_controller_state, prev_heads_states = prev_state

        prev_reads=torch.cat(prev_reads,dim=-1)

        controller_outp, controller_state = self.controller(x,prev_controller_state,delimeter,
                                                            prev_reads=prev_reads,seq=seq)

        reads = []
        heads_states = []
        write_head_nr = 0
        for head, prev_head_state in zip(self.heads, prev_heads_states):
            if head.is_read_head():
                r, head_state = head(controller_outp, prev_head_state, self.num_read_heads)
                reads += [r]

            else:
                if not read_only:
                    head_state = head(controller_outp, prev_head_state, self.num_read_heads)
                    write_head_nr += 1
            heads_states += [head_state]

        ntm_out = torch.cat([controller_outp] + reads, dim=1)
        predictions = self.fc(ntm_out)

        state = (reads, controller_state, heads_states)

        return predictions, state
    
    
    def ntm_save(self,location):
        torch.save(self.fc.state_dict(), location+"decision_layer.pth")
        self.memory.save_memory(location)
        torch.save(self.heads.state_dict(), location+"heads.pth")
        self.controller.save_weights(location)
        

    
    def ntm_laod(self,location):
        self.fc.load_state_dict(torch.load(location+"decision_layer.pth", weights_only=True))
        self.memory.load_memory(location)
        self.controller.load_weights(location)
        self.heads.load_state_dict(location+"heads.pth")
        
