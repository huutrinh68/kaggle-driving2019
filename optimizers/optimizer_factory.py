import torch
from utils.logger import log

def get_optimizer(model, cfg):
    if cfg.optimizer.name == "Adam":
        optimizer = getattr(torch.optim, cfg.optimizer.name)([
            {'params': model.parameters(), 'lr': cfg.optimizer.params.lr},
        ])
    
    log.info('\n')
    log.info('** optim setting **')
    log.info(f'optimizer: {optimizer}')
    return optimizer