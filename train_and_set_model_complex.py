import torch
from sklearn.preprocessing import OneHotEncoder
from aio_complex import EncapsulatedNTM
from torch import nn
from torch.autograd import Variable
from torch import optim
import numpy as np
import cv2 as cv
from collections import deque
from pathlib import PurePath










class MANN_Handler():
    def __init__(self,file_path,batch_size,look_back=3):
        self.criterion = nn.CrossEntropyLoss(reduce=False)
        self.model =EncapsulatedNTM([3,100,100],2,1,controller_size=200, controller_layers=3,
                                num_read_heads=10,num_write_heads=10, N=128, M=40)
        
        self.optimizer = optim.Adam(self.model.parameters())
        self.previous_state=None
        self.load_data_get_vals(file_path)
        self.batch_size=batch_size
        self.look_back=look_back
        self.single=False
        self.type_reward={
                            }
        self.fear_state=deque(maxlen=2)
        self.baseline_resilience=100
        self.exponential_intrisic=0
        self.exponential_rate=.5
        
        
    def load_data_get_vals(self,file_path):
        data_obs = np.load(str(PurePath(file_path+"MiniWorld-Sidewalk-v0_lookback_3observationsfinal_shape.npy")))
        data_class = np.load(str(PurePath(file_path+"MiniWorld-Sidewalk-v0_lookback_3class.npy")))
        data_class_num = np.load(str(PurePath(file_path+"MiniWorld-Sidewalk-v0_lookback_3class_number.npy")))
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
                                            self.input_size,self.input_size))
        if self.single == True:
            self.images=self.images[:,2,:,:]
        
        encoder = OneHotEncoder(sparse_output=False)
        self.one_hot_encoded = encoder.fit_transform(self.data_class.reshape(-1, 1))
        
        self.shuffle_data()
        
        if self.single == True:
            self.images=self.images.reshape(-1,self.batch_size,self.channels,self.input_size)
        else:
            self.images=self.images.reshape(-1,self.look_back,self.batch_size,
                                            self.channels,self.input_size,self.input_size)
            
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
        batch_size = 2
        classes=2
        losses, accuracies = [], []
        image_batch=self.images
        label_batch=self.one_hot_encoded
        classes_batch=self.data_class

        for epoch in range(0, int(150)):

            episode_loss = 0.0
            episode_correct = 0.0
            
            model.init_sequence(2)
    
            initial_state = []
            for i in range(2):
                initial_state.append([0 for c in range(classes)])
    
            accum_loss = Variable(torch.zeros(2).type(torch.FloatTensor))
            previous_state=None

            for i_e in range(self.batches):
                images_t, true_labels,true_classes = Variable(torch.tensor(image_batch[i_e]).to(torch.float32)), Variable(torch.tensor(label_batch[i_e])),Variable(torch.LongTensor(classes_batch[i_e]))
                images_t=torch.tensor(images_t)

                delimiter = torch.FloatTensor(initial_state)
                predicted_labels, previous_state = model(x=images_t,
                                                         delimeter=Variable(delimiter),
                                                         previous_state=previous_state)

               
                predicted_indexes = predicted_labels.data.max(1)[1].view(2)
                timestep_correct = sum(predicted_indexes.eq(true_classes.data.view(2)))
                episode_correct += timestep_correct

                initial_state = []
                for b in range(self.batch_size):
                    true_class = true_classes.data[b]
                    initial_state.append([1 if c == true_class else 0 for c in range(classes)])
    
 
                loss = criterion(predicted_labels, true_labels)
                accum_loss += loss
    
                episode_loss += torch.mean(loss).data
    
    
            mean_loss = torch.mean(accum_loss)
    
            optimizer.zero_grad()
    
            mean_loss.backward()
    
            optimizer.step()
            accuracies.append(float((100.0*episode_correct)/(2*self.batches)))
            losses.append(episode_loss)
    
            print("\n\n--- Epoch " + str(epoch) + ", Episode " + str((epoch + 1)*2) + " Statistics ---")
            print("Instance\tAccuracy")    
            print(episode_correct)

        self.model=model            
        return  losses, accuracies,previous_state
   
    def pre_process_imgs(self,image_deque):
        lookback,channel,dim1,dim2=image_deque.shape
        image_deque=np.reshape(image_deque,(lookback*channel,
                                                    dim1,dim2))
        

        look_back_images=np.asarray([cv.resize(img, (self.input_size, self.input_size),
                                       interpolation=cv.INTER_AREA) for img in image_deque])
        reshaped=np.reshape(look_back_images,(self.look_back,
                                              self.channels,
                                              self.input_size,
                                              self.input_size))

        return reshaped
    
    
    def just_inference(self,img):
        img=self.pre_process_imgs(img)
        hold=np.zeros((self.look_back,
                       self.channels,
                       self.input_size,
                       self.input_size),dtype="float32")
        
        img=np.stack((img,hold))
        img=torch.tensor(img)
        look_back_state = img.view(self.look_back,
                                   self.batch_size,
                                   self.channels,
                                   self.input_size,
                                   self.input_size)
        initial_state = []
        for i in range(2):
            initial_state.append([0 for c in range(2)])
        initial_state=torch.FloatTensor(initial_state)
        delimiter = torch.FloatTensor(initial_state)
        previous_state=None
        
        prediction,_ = self.model(x=look_back_state,
                                 # delimeter=Variable(delimiter),
                                 delimeter=delimiter,
                                 previous_state=self.previous_state)

        return prediction[0]

    def calculate_reward(self,img):
        with torch.no_grad():
            prediction=torch.softmax(self.just_inference(img),dim=-1)
        data= np.asarray(prediction.data)
        reward=data[0]/1
        return reward
    
    # first type of reward non contius reward 
    
    def calculate_threshholded_fear(self,img,threshold=.5):
        with torch.no_grad():
            prediction=torch.softmax(self.just_inference(img),dim=-1)
        data= np.asarray(prediction.data)
        if data[0]>threshold:
            reward=data[0]
        else:
            reward=0
        return reward
        
        
