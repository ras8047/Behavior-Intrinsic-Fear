import gym
from typing import List, Optional
import gym_miniworld
import numpy as np
import cv2






class GymEnvManager:
    def __init__(self,env_name: str, args=None,seed=12345,wrappers=None):
        if args==None:
            self.env_name = env_name
            self.wrappers = wrappers 
            self.env = self._create_env()
            self.seed=seed
        else:
            self.args=args
            self.channel,self.height,self.width=self.args.env_shape
            self.env_name=self.args.env_name
            self.seed=self.args.seed
            self.env = self._create_env()
        cv2.startWindowThread()
        cv2.namedWindow('Window')
            
    def _create_env(self):
        env = gym.make(self.env_name,max_episode_steps=150)
        # env=RTimeLimit(env,400)
        return env

    def reset(self):
        if self.args is not None :
            state=self.env.reset()
            
            state = cv2.resize(state, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return np.reshape(state, (3,self.width,self.height))
            
        if self.seed is not None:
            state=self.env.reset()
            state = cv2.resize(state, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return np.reshape(state, (3,self.width,self.height))
        return self.env.reset()

    def step(self, action):
        if self.args is not None :
            state,reward,done,terminal,info=self.env.step(action)
            state = cv2.resize(state, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return np.reshape(state, (3,self.width,self.height)),reward,done,terminal,info
        return self.env.step(action)


    def close(self):

        self.env.close()



