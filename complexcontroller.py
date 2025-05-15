import torch
import torch.nn.functional as f
from torch import nn
from torch.autograd import Variable
import torch.autograd as autograd






class ComplexLSTMController():
    
    def __init__(self,image_input,prev_reads,
                 hidden_size,multi_input=True,number_layer=3):
        super(ComplexLSTMController, self).__init__()

        self.image_input=image_input
        self.num_outputs = hidden_size
        self.prev_reads=prev_reads
        self.hidden_size=hidden_size
        self.multi_input = multi_input
        self.number_layer=number_layer


        self.complexlstm=nn.ModuleList([ComplexLSTMCell(image_input,prev_reads,
                                                        hidden_size) for i in range(number_layer)])
        
    def create_new_state(self,batch_size):
        
        lstm_h = autograd.Variable(torch.zeros(batch_size, self.num_outputs))
        lstm_c = autograd.Variable(torch.zeros(batch_size, self.num_outputs))
        return lstm_h, lstm_c
        
    def size(self):
        return self.image_input[0], self.num_outputs
        
    def __call__(self,x, prev_state,delimeter, prev_reads=None, class_vector=None, seq=3):

        if self.multi_input or self.number_layer>1:
            if prev_state==None:
                lstm_h,lstm_c=self.create_new_state(len(x[1]))
            else:
                lstm_h,lstm_c=prev_state
            
            for i in range(seq):
                # print(f"this is the shape of a single input to complex lstm {x[i].shape}")
                for index,net in enumerate(self.complexlstm):
                    # lstm_h[index],lstm_c[index]=net(x[i],[lstm_h[index],lstm_c[index]],prev_reads,delimeter)
                    lstm_h,lstm_c=net(x[i],[lstm_h,lstm_c],prev_reads,delimeter)


            # out,state=lstm_h[-1],[lstm_h,lstm_c]
            out,state=lstm_h,[lstm_h,lstm_c]
        
        else:
            out, state = self.complexlstm(x,prev_state,prev_reads)

        return out,state

    def save_weights(self,location):
        torch.save(self.complexlstm.state_dict(),location+"controller_weights.pth")
        
        
    def load_weights(self,location):
        self.complexlstm.load_state_dict(location+"controller_weights.pth")
        


class SiameseNetwork(nn.Module):
    def __init__(self, image_input_shape, vector_input_dim2,vector_input_dim,
                                             embedding_dim=200,output_dim=300):
        super(SiameseNetwork, self).__init__()
        
        # Image branch: Convolutional layers
        self.image_branch = nn.Sequential(
            nn.Conv2d(in_channels=image_input_shape[0], out_channels=10, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=100, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=100, out_channels=250, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        
        # Calculate the flattened size after the conv layers
        self.flattened_size = self._get_flattened_size(image_input_shape)
        
        self.image_fc = nn.Sequential(
             nn.Linear(400002 , embedding_dim),#the 2 is for the delimeter
             nn.ReLU()
         )

        # Vector branch: Fully connected layers
        self.vector_branch = nn.Sequential(
            nn.Linear(vector_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        
        
        self.vector_branch2 = nn.Sequential(
            nn.Linear(vector_input_dim2, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        
        # Final layer to calculate similarity or classification based on combined embeddings
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim+embedding_dim+embedding_dim,output_dim), 
            nn.ReLU() 
        )

    def _get_flattened_size(self, image_input_shape):
        # Forward a dummy input through conv layers to calculate output shape
        with torch.no_grad():
            x = torch.zeros(1, *image_input_shape)
            x = self.image_branch(x)
            return x.view(1, -1).size(1)

    def forward(self, image, vector,vector2,delimeter):
        # Process image branch

        image_features = self.image_branch(image)
        image_features = image_features.view(image_features.size(0), -1)  # Flatten
        # print(f"image_features shape is {image_features.shape} and delimeter {delimeter.shape}")
        image_features=torch.cat((image_features,delimeter),dim=-1)
        image_embedding = self.image_fc(image_features)

        # Process vector branch
        # print(f"vector shape is {vector}")
        vector_embedding = self.vector_branch(vector)

        #vector 2 branch
        vector_embedding2 = self.vector_branch2(vector2)
        
        # Combine embeddings
        combined = torch.cat((image_embedding, vector_embedding,vector_embedding2), dim=1)
        # print(combined.size())
        
        # Pass combined embeddings through output layer
        output = self.output_layer(combined)
        return output



class ComplexLSTMCell(nn.Module):

    def __init__(self, image_input,prev_reads,hidden_size):
        super().__init__()
        self.image_input = image_input
        self.prev_reads = prev_reads
        self.hidden_size=hidden_size
        self.Gates = SiameseNetwork([*image_input],prev_reads,
                                    self.hidden_size,
                                    embedding_dim=self.hidden_size,
                                    output_dim=4*hidden_size)
    #x[i],[lstm_h[index],lstm_c[index]],prev_reads,delimeter
    def forward(self, input_, prev_state,prev_read,delimeter):

        # print(f"prev_state is {prev_state[0].shape}")

        # get batch and spatial sizes
        batch_size = input_.data.size()[0]
        spatial_size = input_.data.size()[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] 
            prev_state = (
                Variable(torch.zeros(state_size)),
                Variable(torch.zeros(state_size))
            )

        prev_hidden, prev_cell = prev_state
        # print(f"this is the prev_state im getting prev_state {prev_hidden.shape} ")

        # data size is [batch, channel, height, width]
        # stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(input_,prev_hidden,prev_read,delimeter)

        # chunk across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)

        in_gate = f.sigmoid(in_gate)
        remember_gate = f.sigmoid(remember_gate)
        out_gate = f.sigmoid(out_gate)

        cell_gate = f.tanh(cell_gate)

        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * f.tanh(cell)

        return hidden, cell
