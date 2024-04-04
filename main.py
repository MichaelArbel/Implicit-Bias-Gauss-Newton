
import mlxp
from training import Trainer
import os
import dill as pkl
import time
from omegaconf import DictConfig, OmegaConf
import yaml


def set_seeds(seed):    

    import torch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    import random
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

@mlxp.launch(config_path='./configs',
             seeding_function=set_seeds)
def main(ctx):
    
    try:
        trainer  = ctx.logger.load_checkpoint(log_name='last_ckpt') 
        print("Loading from latest checkpoint")
     
    except:
        print("Failed to load checkpoint, Starting from scratch")
        trainer = Trainer(ctx.config, ctx.logger)


    trainer.train()


if __name__ == "__main__":
    main()


    