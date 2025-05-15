from collections import deque
import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import PurePath


class EpisodeVisualizer:
    
    def __init__(self,steps_per,state_dims,video_name,output_path,visualize_rate=100):
        self.video_name=video_name
        self.output_path=output_path
        self.state_dims=state_dims #channel last 
        self.video_images=deque(maxlen=steps_per)
        self.visualize_rate=visualize_rate
        
    def capture_state(self,state):
        self.video_images.append(state)
        
    def visulize_state(self,state: np.ndarray, title: str = "Image"):

        plt.figure(figsize=(6, 6))
        plt.imshow(state)
        plt.axis('off')
        plt.title(title)
        plt.show(block=True)
        

    def track(self,episode,state,intrinsic_reward):
        channel,height,width=state.shape
        state=state.reshape(height,width,channel)
        if episode % self.visualize_rate==0:
            self.visulize_state(state,title=f"this was the intrinsic reward{intrinsic_reward}")
            
        


class VideoTracker():
    def __init__(self,save_folder):
        self.keeper_img=deque(maxlen=150)
        self.keeper_intrinsic=deque(maxlen=150)
        self.save_rate=100
        self.save_folder=save_folder
        
    def append(self,state,intrinsic_reward,done,episode):
        if episode % 100 ==0:
            self.keeper_img.append(state)
            self.keeper_intrinsic.append(intrinsic_reward)
            if done == True:
                # self.save(episode)
                np.save(str(PurePath(self.save_folder+f"\\episode{episode}_states")),
                                                    np.asarray(self.keeper_img))
                np.save(str(PurePath(self.save_folder+f"\\instrinsic_reward_at_{episode}")),
                                                np.asarray(self.keeper_intrinsic))
                self.keeper_img.clear()
                self.keeper_intrinsic.clear()
            
            
            
    def save(self,episode):
        images=np.asarray(self.keeper_img)
        font=cv2.FONT_HERSHEY_SIMPLEX
        font_scale=1
        color=(0, 255, 0)
        thickness=2
        _,height, width, _ = images.shape

        writer_setup = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(f"saved_episode{episode}", writer_setup, 30, (width, height))
    
        
        for i, frame in enumerate(self.keeper_img):
            out.write(np.uint8(frame))  
            position = (50, height - 50)
            cv2.putText(frame, f"the instrinsic reward was{self.keeper_intrinsic[i]} ", position, font, font_scale, color, thickness, cv2.LINE_AA)
            out.write(frame)
        
        out.release()