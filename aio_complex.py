import torch
from torch import nn
from torch.autograd import Variable
from ntm_complex import NTM
from complexcontroller import ComplexLSTMController
from head import NTMReadHead, NTMWriteHead
from memory import NTMMemory







class EncapsulatedNTM(nn.Module):

    def __init__(self, num_inputs, num_outputs, num_classes,
                 controller_size, controller_layers, num_read_heads, num_write_heads, N, M):

        super(EncapsulatedNTM, self).__init__()
        self.num_heads = num_read_heads + num_write_heads
        self.controller_layers = controller_layers
        self.controller_size = controller_size
        self.num_write_heads = num_write_heads
        self.num_read_heads = num_read_heads
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.N = N
        self.M = M
        self._set_components()
        
        
    def _set_components(self):
        memory = NTMMemory(self.N, self.M)
                                          #(image_input,prev_reads,hidden_size,multi_input=True,number_layer=3)
        controller = ComplexLSTMController(self.num_inputs,self.M*self.num_read_heads
                                           ,self.controller_size,number_layer=self.controller_layers)
        heads = nn.ModuleList([])
        for i in range(self.num_read_heads):
            heads += [
                NTMReadHead(memory, self.controller_size),
            ]
        for i in range(self.num_write_heads):
            heads += [
                NTMWriteHead(memory, self.controller_size)
            ]
        self.ntm = NTM(self.num_outputs, controller, memory, heads)
        self.memory = memory


    def init_sequence(self, batch_size):
        self.batch_size = batch_size
        self.memory.reset(batch_size)
        self.previous_state = self.ntm.create_new_state(batch_size)
        return self.previous_state

    def forward(self, x=None,delimeter=None, previous_state=None, class_vector=None, read_only=False,seq=1):
        if x is None:
            x=self.zero_pad()

        if previous_state == None:
            o, self.previous_state = self.ntm(x, self.previous_state,delimeter, class_vector=class_vector, read_only=read_only,seq=seq)
            return o, self.previous_state
        else:
            return self.ntm(x, previous_state,delimeter, class_vector=class_vector, read_only=read_only)

    def calculate_num_params(self):
        num_params = 0
        for p in self.parameters():
            num_params += p.data.view(-1).size(0)
        return num_params
    
    def zero_pad(self):
        x = Variable(torch.zeros(self.batch_size, self.num_inputs))
        return x

    def save(self,location):
        self.ntm.ntm_save(location)
        
        
    def load(self,location):
        self.ntm.ntm_load(location)
    
    


