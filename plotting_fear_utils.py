import matplotlib.pyplot as plt
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator






def moving_average(data, window_size=30):
    kernel = np.ones(window_size) / window_size
    smoothed_data = np.convolve(data, kernel, mode='same') 

    return smoothed_data


def load_tensorboard_data(log_dirs: list, tags: list, title: str):
    runs_data = {}
    all_runs = []
    
    if not os.path.exists("Graphs"):
        os.makedirs("Graphs")
        
    for i, log_dir in enumerate(log_dirs, start=1):
        event_acc = EventAccumulator(log_dir)
        event_acc.Reload()
        
        run_data = {}
        for tag in tags:
            if tag in event_acc.Tags()["scalars"]:
                events = event_acc.Scalars(tag)
                run_data[tag] = moving_average([event.value for event in events])
        
        runs_data[f"run_{i}"] = run_data
        all_runs.append(run_data)
    
    avg_data = {}
    std_data={}
    for tag in tags:
        tag_values = [run[tag] for run in all_runs if tag in run]
        if tag_values:
            avg_data[tag] = np.mean(tag_values, axis=0).tolist()
            std_data[tag] = np.std(tag_values, axis=0).tolist()
            
    
    runs_data["average"] = avg_data
    
    for tag in tags:
        plt.figure()
        for i, run in enumerate(all_runs, start=1):
            if tag in run:
                plt.plot(run[tag], label=f"Run {i}")
        if tag in avg_data:
            plt.plot(avg_data[tag], label="Average", linestyle='dashed', linewidth=1, color='black',alpha=.7)
            plt.fill_between(range(len(avg_data[tag])),np.asarray(avg_data[tag])-np.asarray(std_data[tag]),np.asarray(avg_data[tag])+np.asarray(std_data[tag]),alpha=.3,color='black')
        true_title=f"{title} {tag}".replace("/", "")
        plt.title(true_title)
        plt.xlabel("Episodes")
        plt.ylabel(tag.replace("charts/", ""))
        # plt.legend(loc='lower left')
        plt.legend(loc='lower left')
        plot_path = os.path.join("Graphs",true_title+".jpeg")
        plt.savefig(plot_path)
        plt.show()

    
    return runs_data



def iterloadboardthresh(list_vals,
                  tags=['charts/episode_lengh', 'charts/intrinsic_rewards','charts/episodic_rewards','charts/joint_rewards'],
                  main_title="MiniWorld Results with Expanded Controller and thresholded"):
    all_values=[]

    for thresh in list_vals:
        main_folders=['first_mann_runcomplexthreshhold', 
                  'second_mann_runcomplexthreshhold',
                  'third_mann_runcomplexthreshhold',
                  'fourth_mann_runcomplexthreshhold',
                  'fifth_mann_runncomplexthreshhold']
        mod_folders = list(map(lambda x: x + thresh, main_folders))
        mod_title=main_title+thresh
        all_values.append(load_tensorboard_data(mod_folders,tags,mod_title))
    return all_values



def iterloadboardbase(list_vals,
                  tags=['charts/episode_lengh', 'charts/intrinsic_rewards','charts/episodic_rewards','charts/joint_rewards'],
                  main_title="MiniWorld Results with No Intrinsic"):
    all_values=[]

    for thresh in list_vals:
        main_folders=['first_mann_runno_intrinsic', 
                  'second_mann_runno_intrinsic',
                  'third_mann_runno_intrinsic',
                  'fourth_mann_runno_intrinsic',
                  'fifth_mann_runnno_intrinsic']
        mod_folders = list(map(lambda x: x + thresh, main_folders))
        mod_title=main_title+thresh
        all_values.append(load_tensorboard_data(mod_folders,tags,mod_title))
    return all_values
        
        
def iterloadboardnointrin(list_vals,
                  tags=['charts/episode_lengh', 'charts/intrinsic_rewards','charts/episodic_rewards','charts/joint_rewards'],
                  main_title="MiniWorld Results with Stimuli Intrinsic"):
    all_values=[]
    for thresh in list_vals:
        main_folders=['first_mann_runnormalbase', 
                  'second_mann_runnormalbase',
                  'third_mann_runnormalbase',
                  'fourth_mann_runnormalbase',
                  'fifth_mann_runnnormalbase']
        mod_folders = list(map(lambda x: x + thresh, main_folders))
        mod_title=main_title+thresh
        all_values.append(load_tensorboard_data(mod_folders,tags,mod_title))
    return all_values



def extract_value(nested_dict,nest_type=None,nest_vals=None):
    list_vals=["0.25","0.3","0.35","0.4","0.45","0.5","0.55","0.6","0.65","0.7","0.75","0.8","0.85","0.9","0.95"]
    if nest_type=="thresh":
        all_values=[]
        for i in nest_vals:
            dict_index=list_vals.index(i)
            print(dict_index)
            intrinsic=nested_dict[dict_index]["average"]["charts/intrinsic_rewards"]
            extrinsic=nested_dict[dict_index]["average"]["charts/episodic_rewards"]
            episode_lengh=nested_dict[dict_index]["average"]["charts/episode_lengh"]
            avg_intrinsic=np.average(intrinsic)
            avg_extrinsic=np.average(extrinsic)
            avg_episode_lengh=np.average(episode_lengh)
            std_intrinsic=np.std(intrinsic)
            std_extrinsic=np.std(extrinsic)
            std_episode_lengh=np.std(episode_lengh)
            all_values.append(f"{list_vals[dict_index]}, avg_intrinsic={avg_intrinsic} ,std_intrinsic={std_intrinsic}, avg_extrinsic={avg_extrinsic},std_extrinsic={std_extrinsic} ,avg_episode_lengh={avg_episode_lengh} ,std_episode_lengh={std_episode_lengh}")
        return all_values
    else:
        dict_index=0
        intrinsic=nested_dict[dict_index]["average"]["charts/intrinsic_rewards"]
        extrinsic=nested_dict[dict_index]["average"]["charts/episodic_rewards"]
        episode_lengh=nested_dict[dict_index]["average"]["charts/episode_lengh"]
        avg_intrinsic=np.average(intrinsic)
        avg_extrinsic=np.average(extrinsic)
        avg_episode_lengh=np.average(episode_lengh)
        std_intrinsic=np.std(intrinsic)
        std_extrinsic=np.std(extrinsic)
        std_episode_lengh=np.std(episode_lengh)
        return f"{nest_vals}, avg_intrinsic={avg_intrinsic} ,std_intrinsic={std_intrinsic}, avg_extrinsic={avg_extrinsic},std_extrinsic={std_extrinsic} ,avg_episode_lengh={avg_episode_lengh} ,std_episode_lengh={std_episode_lengh}"
    
    
    
list_vals=["0.25","0.3","0.35","0.4","0.45","0.5","0.55","0.6","0.65","0.7","0.75","0.8","0.85","0.9","0.95"]
all_vals_thresh=iterloadboardthresh(list_vals)
list_vals=[""]
all_vals_base_intrincsic=iterloadboardnointrin(list_vals)
list_vals=[""]
all_vals_base=iterloadboardbase(list_vals)


table_vals_base=extract_value(all_vals_base)
table_vals_stimuli=extract_value(all_vals_base_intrincsic)
table_vals_thresh_val=extract_value(all_vals_thresh,"thresh",["0.25","0.6","0.9"])