from torch.utils.tensorboard import SummaryWriter
import os






class TLogger:
    def __init__(self,args,metrics={}):
        self.log_dir = args.save_folder
        os.makedirs(self.log_dir, exist_ok=True) 
        self.summary_writer = SummaryWriter(self.log_dir)
        self.metrics = metrics
        self._step_counter = 0
        

    def ad_key(self,new_key):
        self.metrics[new_key]=True
        
    def log(self, data):
        for i in data:
            key,value=i
            if key in self.metrics:
                self.summary_writer.add_scalar("charts/"+key, float(value), global_step=self._step_counter)
            else:
                self.ad_key(key)
                self.summary_writer.add_scalar("charts/"+key, float(value), global_step=self._step_counter)
                
        self._step_counter += 1





        
     
        
        
        
    
    
    