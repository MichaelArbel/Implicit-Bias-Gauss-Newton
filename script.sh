#!/bin/bash










########### Figure 2 ##########################

device=-1
dtype=32
seed=1
parent_work_dir="./data/.workdir" 
parent_log_dir="./data/outputs/figure_2" 


### GD and GN
HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb main.py  \
                system.device=$device\
                system.dtype=$dtype\
                seed=1,2,3,4,5\
                optimizer.use_GN=True,False\
                optimizer.lr=10.\
                init.pre_train=False\
                init.std=0.01,0.1,0.5,1.,5.,10.,100.\
                +mlxp.interactive_mode=True\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.use_version_manager=True\

## Random features
HYDRA_FULL_ERROR=1   OC_CAUSE=1 python -m ipdb main.py  \
                system.device=$device\
                system.dtype=$dtype\
                seed=1,2,3,4,5\
                optimizer.use_GN=True\
                optimizer.lr=10.\
                init.pre_train=True\
                init.std=0.01,0.1,0.5,1.,5.,10.,100.\
                optimizer.max_iter=0\
                init.pre_train_iter=1000\
                +mlxp.interactive_mode=True\
                +mlxp.version_manager.parent_work_dir=$parent_work_dir\
                +mlxp.logger.parent_log_dir=$parent_log_dir\
                +mlxp.use_version_manager=True\












