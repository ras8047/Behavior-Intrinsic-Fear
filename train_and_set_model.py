from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
from torch.autograd import Variable
from torch import optim
from torch import nn
from aio import EncapsulatedNTM
import cv2 as cv







class MANN_Handler():
    def __init__(self,file_path,batch_size,look_back=3):
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.model = EncapsulatedNTM(1600 + 2, 2,1,controller_size=200, controller_layers=3, num_read_heads=10,num_write_heads=10, N=128, M=40)
        self.optimizer = optim.Adam(self.model.parameters())
        self.previus_state=None
        self.load_data_get_vals(file_path)
        self.batch_size=batch_size
        self.look_back=look_back
        self.single=True
        
    def load_weights(self):
        self.previus_state=torch.load("previous_state")
        
    def load_data_get_vals(self,file_path):
        data_obs = np.load(file_path+"MiniWorld-Sidewalk-v0_lookback_3observationsfinal_shapebw_shape.npy")
        data_class = np.load(file_path+"MiniWorld-Sidewalk-v0_lookback_3class.npy")
        data_class_num = np.load(file_path+"MiniWorld-Sidewalk-v0_lookback_3class_number.npy")
        self.num_data=data_obs.shape[0]
        self.look_back=data_obs.shape[1]
        self.channels=data_obs.shape[2]
        self.input_size=data_obs.shape[3]
        
        self.images = data_obs
        self.data_class = data_class_num
        
    
    def preprocess_data(self):
        self.images=np.reshape(self.images,(-1,
                                            self.look_back,
                                            self.channels,
                                            self.input_size))
        if self.single == True:
            self.images=self.images[:,2,:,:]
        
        encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_encoded = encoder.fit_transform(self.data_class.reshape(-1, 1))
        
        self.shuffle_data()
        
        if self.single == True:
            self.images=self.images.reshape(-1,self.batch_size,self.channels,self.input_size)
        else:
            self.images=self.images.reshape(-1,self.batch_size,self.input_size)
            
        self.batches=self.images.shape[0]

        self.one_hot_encoded=self.one_hot_encoded.reshape(self.batches,self.batch_size,2)
        self.data_class=self.data_class.reshape(self.batches,self.batch_size,-1)
    
        
    def shuffle_data(self):
        
        randomize = np.arange(len(self.data_class))
        np.random.shuffle(randomize)
        self.data_class = self.data_class[randomize]
        self.images = self.images[randomize]
        self.one_hot_encoded= self.one_hot_encoded[randomize]

    
    
    
    def train_model(self):
        optimizer=self.optimizer
        criterion=self.criterion
        model=self.model
        self.preprocess_data()
        classes=2
        losses, accuracies = [], []
        image_batch=self.images
        label_batch=self.one_hot_encoded
        classes_batch=self.data_class

        for epoch in range(0, int(600/2)):

            #TRACKING
            episode_loss = 0.0
            episode_correct = 0.0
            
            model.init_sequence(self.batch_size)
    
            # Creating initial state:
            initial_state = []
            for i in range(self.batch_size):
                initial_state.append([0 for c in range(classes)])
    
            # Placeholder for loss variable:
            accum_loss = Variable(torch.zeros(2).type(torch.FloatTensor))
            previous_state=None
            for i_e in range(self.batches):
                # Colelcting timestep images + labels:
                images_t, true_labels,true_classes = image_batch[i_e], Variable(torch.tensor(label_batch[i_e])),Variable(torch.LongTensor(classes_batch[i_e]))
                images_t=torch.tensor(images_t)
                # Flattening images:
                if self.single:
                    flattened_images_t = images_t.view(self.batch_size,-1)
                else:
                    flattened_images_t = images_t.view(self.batch_size,
                                                       self.look_back, -1)
    
                initial_state = torch.FloatTensor(initial_state)

                state = torch.cat((flattened_images_t, initial_state), 1)
    
                # feed the sequence + delimiter
                predicted_labels, previous_state = model(x=Variable(state),previous_state=previous_state)

                predicted_indexes = predicted_labels.data.max(1)[1].view(2)

                timestep_correct = sum(predicted_indexes.eq(true_classes.data.view(2)))
                episode_correct += timestep_correct

                initial_state = []
                for b in range(self.batch_size):
                    true_class = true_classes.data[b]
                    initial_state.append([1 if c == true_class else 0 for c in range(classes)])
    
 

                #calculate the loss
                loss = criterion(predicted_labels, true_labels)
                accum_loss += loss
    
                episode_loss += torch.mean(loss).data
    
 
            mean_loss = torch.mean(accum_loss)
    
            
            #perform optimization step
            optimizer.zero_grad()
            mean_loss.backward()
            optimizer.step()
            
            
            accuracies.append(float((100.0*episode_correct)/(self.batch_size*self.batches)))
            losses.append(episode_loss)
    
            print("\n\n--- Epoch " + str(epoch) + ", Episode " + str((epoch + 1)*2) + " Statistics")
            print(episode_correct)
        self.model=model
        return  losses, accuracies,previous_state
    
    def pre_process_img(self,img):
        _,channel,dim1,dim2=img.shape
        img=np.reshape(img,(dim1,dim2,channel))
        resized=np.asarray(cv.resize(img, (40,40), interpolation = cv.INTER_AREA))
        reshaped=np.reshape(resized,(40,40,3))
        grayImage = cv.cvtColor(reshaped, cv.COLOR_BGR2GRAY)
        return grayImage
    
    
    def just_inference(self,img):
        img=self.pre_process_img(img)
        hold=np.zeros((40,40),dtype="float32")
        img=np.stack((img,hold))
        img=torch.tensor(img)
        flattened_images_t = img.view(2, -1)
        initial_state = []
        for i in range(2):
            initial_state.append([0 for c in range(2)])
        initial_state=torch.FloatTensor(initial_state)
        state = torch.cat((flattened_images_t, initial_state), 1)
        with torch.no_grad():
            prediction,_=self.model(x=(state),previous_state=self.previus_state,read_only=True)
        return prediction[0]

    def calculate_reward(self,img):
        prediction=torch.softmax(self.just_inference(img),dim=-1)
        data= np.asarray(prediction.data)
        similarity=data[0]
        return similarity
