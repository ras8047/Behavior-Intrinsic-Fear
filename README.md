# Behavior-Intrinsic-Fear# Avoiding Death through Fear Intrinsic Conditioning

This repository is the official implementation of [Avoiding Death through Fear Intrinsic Conditioning](). 


To install requirements:

First, download the env tar at:

```
conda install conda-pack
```

Download the conda pack at https://drive.google.com/drive/folders/1Utb8bnv8lPfoLFRcj4HfFlTqJ6xwo54n?usp=sharing

Then follow the proper unpack conditions for you OS the original work was done on WINDOWS all conda unpack instrunction found 
here for linux :https://conda.github.io/conda-pack/
here for windows: https://gist.github.com/pmbaumgartner/2626ce24adb7f4030c0075d2b35dda32

After unpacking and activating the environment, Run the following for each specific Test:

To Run the LOW Threshold Test Use the following command line:

'''
python miniworldtrainer.py  --intrisic-run  --which_mann complex --intrinsic_type threshhold --threshold .25 .30 .35 .40 .45 
'''

To Run the MID Threshold Test Use the following command line:

'''
python miniworldtrainer.py  --intrisic-run  --which_mann complex --intrinsic_type threshhold --threshold .50 .55 .60 .65 .70 
'''


To Run the High Threshold Test Use the following command line:

'''
python miniworldtrainer.py  --intrisic-run  --which_mann complex --intrinsic_type threshhold --threshold .75 .80 .85 .90 .95 
'''


To Run the Stimuli Test Use the following command line:

'''
python miniworldtrainer.py --intrisic-run --which-mann normal --intrinsic_type base
'''

To Run the Base PPO Test Use the following command line:

'''
python miniworldtrainer.py --intrisic-run --which-mann normal --intrinsic_type base
'''
IF COMMAND LINE arguements do no work please open miniworldtrainer.py  and uncommnent the predifined RunArgs Testing parameters 

Using the Generated logs use the plotting_fear_utils.py 
To Generate the Graphs and Table values


