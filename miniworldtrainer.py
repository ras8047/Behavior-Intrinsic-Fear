from ppoagenthandler import PPO as Agent
from gym_handler import GymEnvManager as EnvHandler
from loggerutils import TLogger
from dataclasses import dataclass, field
from train_and_set_model import MANN_Handler
from train_and_set_model_complex import MANN_Handler as MANN_HandlerComplex 
from videoutils import EpisodeVisualizer,VideoTracker
from typing import List
import tyro
import os 
import numpy as np
from collections import deque
from pathlib import PurePath






# data_class for environment
@dataclass
class EnvArgs:
    env_name:str='MiniWorld-Sidewalk-v0' 
    """environment ID"""
    seed:int= np.random.randint(100000, 999999)
    env_shape:tuple=(3,84,84)
    
# data_class for agent
@dataclass
class AgentArgs:
    lr_actor:float= 1e-5
    """learning rate for actor """
    lr_critic:float= 1e-5
    """learning rate for actor """
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 1 # NOTE: Changed from 8000 to 1
    """the frequency of updates for the target networks"""
    eps_clip:float = 0.3  
    "the clipping for kl for PPO"
    k_epochs: int = 40
    """number of epochs train at every step"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""

# data_class for logger_params
@dataclass
class LoggerArgs:
    exp_name:List[str] =field(default_factory=lambda:["first_mann_run",
                                                      "second_mann_run",
                                                      "third_mann_run",
                                                      "fourth_mann_run",
                                                      "fifth_mann_runn"])
    # exp_name:str= 'second_mann_run'
    """experiment name"""
    controller_type:str ="_simple_controller"
    """descrives the controller used in the experiment"""
    test_condition:str="baselines"
    """contion which this experiment test"""
    save_folder:str="logs\\"


# #Test No Intrinsic Params
# @dataclass
# class RunArgs:
#     intrisic_run:bool=False
#     """start-level for environment"""
#     max_epis_len: int = 150
#     """the max amount of steps for any episode"""
#     episodes:int=1000
#     """the total number of episodes"""
#     update_timestep: int = max_epis_len*5
#     """the frequency of training updates"""
#     look_back:int = 3
#     """number of states to preserve for agent to give values for """
#     single:bool = False
#     """number of states to preserve for agent to give values for """
#     which_mann:str="normal"
#     """ descrives which controller was used """
#     baseline_flag:bool=False
#     """flag that sets baseline resilience method """
#     intrinsic_type:str="base"
#     """ set the type of intrinsic reward to use for assignment """
#     threshold:List[float] =field(default_factory=lambda:[.50,.55,.60,.65,.70])




# #Test stimuli params
# @dataclass
# class RunArgs:
#     intrisic_run:bool=True
#     """start-level for environment"""
#     max_epis_len: int = 150
#     """the max amount of steps for any episode"""
#     episodes:int=1000
#     """the total number of episodes"""
#     update_timestep: int = max_epis_len*5
#     """the frequency of training updates"""
#     look_back:int = 3
#     """number of states to preserve for agent to give values for """
#     single:bool = False
#     """number of states to preserve for agent to give values for """
#     which_mann:str="normal"
#     """ descrives which controller was used """
#     baseline_flag:bool=False
#     """flag that sets baseline resilience method """
#     intrinsic_type:str="base"
#     """ set the type of intrinsic reward to use for assignment """
#     threshold:List[float] =field(default_factory=lambda:[.50,.55,.60,.65,.70])

#TEST Threshold params
@dataclass
class RunArgs:
    intrisic_run:bool=True
    """start-level for environment"""
    max_epis_len: int = 150
    """the max amount of steps for any episode"""
    episodes:int=1000
    """the total number of episodes"""
    update_timestep: int = max_epis_len*5
    """the frequency of training updates"""
    look_back:int = 3
    """number of states to preserve for agent to give values for """
    single:bool = False
    """number of states to preserve for agent to give values for """
    which_mann:str="complex"
    """ descrives which controller was used """
    baseline_flag:bool=False
    """flag that sets baseline resilience method """
    intrinsic_type:str="threshhold"
    """ set the type of intrinsic reward to use for assignment """
    threshold:List[float] =field(default_factory=lambda:[.50,.55,.60,.65,.70])


class miniworltraininer():
    
    def __init__(self,iterator,thresh_index=None,RunArgs=RunArgs,AgentArgs=AgentArgs,
                             LoggerArgs=LoggerArgs,EnvArgs=EnvArgs):
        self.env_params=tyro.cli(EnvArgs)
        self.agent_params=tyro.cli(AgentArgs)
        self.logger_params=tyro.cli(LoggerArgs)
        self.run_params=tyro.cli(RunArgs)
        self.thresh_index=thresh_index
        if not self.run_params.intrisic_run:
            self.logger_params.save_folder=self.logger_params.save_folder+self.logger_params.exp_name[iterator]+"no_intrinsic"
            
        elif self.run_params.which_mann=="normal":
            self.logger_params.save_folder=self.logger_params.save_folder+self.logger_params.exp_name[iterator]+self.run_params.which_mann+self.run_params.intrinsic_type
        elif self.run_params.which_mann=="complex":
            self.logger_params.save_folder=self.logger_params.save_folder+self.logger_params.exp_name[iterator]+self.run_params.which_mann+self.run_params.intrinsic_type+str(self.run_params.threshold[self.thresh_index])

        self._set_up_test()
        self.dictionay_intrinsic={"base":self._get_intrinsic,
                                  "threshhold":self._get_intrinsic_thresholded,
                                  "look_away":self._get_intrinsic_look_away}
    
    def _set_up_test(self):
        self.env=EnvHandler(None,self.env_params)
        self.agent=Agent(self.env.env,args=self.agent_params)
        self.logger=TLogger(self.logger_params) 
        self.visualizer=EpisodeVisualizer(100,(self.env_params.env_shape[1],
                                               self.env_params.env_shape[2],
                                               self.env_params.env_shape[0]),
                                               "agent_performance",
                                               self.logger_params.save_folder) 
        
        self.intrinsic_type=self.run_params.intrinsic_type
        self.exporter=VideoTracker(self.logger_params.save_folder)
        
    def _set_intrinsic(self):
        if self.run_params.intrisic_run ==True:
            if self.run_params.which_mann == "normal":

                self.handler=MANN_Handler("Data//prototype//",2)
                losses, accuracies,previous_state=self.handler.train_model()
                # self.handler.model.save(self.logger_params.save_folder+"//weights")
                np.save(str(PurePath(self.logger_params.save_folder+"//losses")),
                                                            losses)
                np.save(str(PurePath(self.logger_params.save_folder+"//accuracies")),accuracies)
                self.mann_states=deque(maxlen=1)
                
            else:
                
                self.handler=MANN_HandlerComplex("Data//prototype//",2)
                if os.path.exists(str(PurePath(self.logger_params.save_folder+"//weights"))):
                    self.handler.model.load(str(PurePath(self.logger_params.save_folder+"//weights")))
                else:
                    losses, accuracies,previous_state=self.handler.train_model()
                    self.handler.model.save(str(PurePath(self.logger_params.save_folder+"//weights")))
                    np.save(str(PurePath(self.logger_params.save_folder+"//losses")),losses)
                    np.save(str(PurePath(self.logger_params.save_folder+"//accuracies")),accuracies)
                self.mann_states=deque(maxlen=3)
    
    def _get_intrinsic(self,global_step,state):
        if self.run_params.intrisic_run ==True:
            self.mann_states.append(state)
            if self.run_params.which_mann=="normal":
                if global_step >0:

                    state_mann=np.asarray(self.mann_states)
                    
                    intrinsic_reward=-1*(self.handler.calculate_reward(state_mann))
            else:
                if global_step >2:
                    state_mann=np.asarray(list(self.mann_states))
                    
                    intrinsic_reward=-1*(self.handler.calculate_reward(state_mann))
                else:
                    intrinsic_reward=0
                    
        else:
            intrinsic_reward=0
        return intrinsic_reward
    
    def _get_intrinsic_thresholded(self,global_step,state):
        if self.run_params.intrisic_run ==True:
            self.mann_states.append(state)
            if self.run_params.which_mann=="normal":
                if global_step >0:

                    state_mann=np.asarray(self.mann_states)
                    
                    intrinsic_reward=-1*(self.handler.calculate_threshholded_fear(state_mann))
            else:
                if global_step >2:
                    state_mann=np.asarray(list(self.mann_states))
                    
                    intrinsic_reward=-1*(self.handler.calculate_threshholded_fear(state_mann))
                else:
                    intrinsic_reward=0
                    
        else:
            intrinsic_reward=0
        return intrinsic_reward
        
    def _get_intrinsic_look_away(self,global_step,state):
        if self.run_params.intrisic_run ==True:
            self.mann_states.append(state)
            if self.run_params.which_mann=="normal":
                if global_step >0:

                    state_mann=np.asarray(self.mann_states)
                    
                    intrinsic_reward=-1*(self.handler.calculates_look_away_fear(state_mann,self.run_params.threshold[self.thresh_index]))
            else:
                if global_step >2:
                    state_mann=np.asarray(list(self.mann_states))
                    
                    intrinsic_reward=-1*(self.handler.calculates_look_away_fear(state_mann,self.run_params.threshold[self.thresh_index]))
                else:
                    intrinsic_reward=0
                    
        else:
            intrinsic_reward=0
        return intrinsic_reward
    
    
    def choose_intrinsic(self,global_step,state):
        return self.dictionay_intrinsic[self.intrinsic_type](global_step,state)
    
    def run_test(self):
        self._set_intrinsic()
        global_step=0
        for ep in range(self.run_params.episodes):
            done = False
            terminal= False
            state=self.env.reset()
            episodic_reward=0
            episodic_intrinsic_reward=0
            episode_steps=0
            if self.run_params.baseline_flag:
                baseline_recilience=150
            

            while  (done==False or terminal==False):
                action =self.agent.act(state)
                next_state,reward,done,terminal,info=self.env.step(action)
                global_step=global_step+1
                episode_steps=episode_steps+1
 
                if terminal==True:
                    done=True
                    

                intrinsic_reward=self.choose_intrinsic(global_step,state)
                if self.run_params.baseline_flag:
                    baseline_recilience=baseline_recilience+intrinsic_reward
                self.visualizer.track(ep,state,intrinsic_reward)
                
                joint_reward=reward+intrinsic_reward
                episodic_reward=episodic_reward+reward
                episodic_intrinsic_reward=episodic_intrinsic_reward+intrinsic_reward
                state=next_state
                if done and self.run_params.baseline_flag:
                    joint_reward=joint_reward+baseline_recilience
                self.agent.remember(joint_reward,done)
                if global_step % self.run_params.update_timestep == 0:
                    print("training")
                    loss,ratios,_=self.agent.train()
                
                self.exporter.append(state, intrinsic_reward, done, ep)
                if done:
                    joint_rewards=episodic_reward+episodic_intrinsic_reward
                    log_list=[("intrinsic_rewards",episodic_intrinsic_reward),
                              ("episodic_rewards",episodic_reward),
                              ("joint_rewards",joint_rewards),
                              ("episode_lengh",episode_steps)]
                    report2=f"The current episode is {ep} the joint_rewards is currently {joint_rewards} total_steps {episode_steps}"
                    print(report2)
                    report=f"The current episode is {ep} the episodic reward is currently {episodic_reward} total_steps {episode_steps}"
                    print(report)
  
                    self.logger.log(log_list)#testing for memory prob
                    break
                    
        
        
    def __call__(self):
        self.run_test()
        


if __name__ == "__main__":
    run_args=RunArgs()
    if run_args.which_mann=="normal":
        thresh=None
        for i in range(5):
            trainer=miniworltraininer(i,thresh)
            trainer()
    else:
        for thresh in range(5):
            # thresh=None
            for i in range(5):
                trainer=miniworltraininer(i,thresh)
                trainer()
        

        
        
        