import os
import random
import glob

import pandas as pd
import numpy as np
import torch

from utils.logger import log

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_model(path, model, optimizer=None, device=None):
    if device:
        # ramap to device
        state = torch.load(str(path), map_location=device)
    else:
        # remap everthing onto CPU 
        state = torch.load(str(path), map_location=lambda storage, location: storage)

    print(state)
    exit(0)
    model.load_state_dict(state['state_dict'])
    exit(0)
    if optimizer:
        log.info('loading optim too')
        optimizer.load_state_dict(state['optimizer'])
    else:
        log.info('not loading optimizer')

    model.to(device)

    # detail = state['detail']
    detail = None
    log.info(f'loaded model from {path}')

    return detail


# evaluate meters
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, optim, detail, fold, dirname):
    # path = os.path.join(dirname, 'fold%d_ep%d.pt' % (fold, detail['epoch']))
    os.makedirs(os.path.join(dirname, str(fold)), exist_ok=True)
    path = os.path.join(dirname, str(fold), 'top1.pth')
    torch.save({
        'state_dict': model.state_dict(),
        'optim': optim.state_dict(),
        'detail': detail,
    }, path)


def get_lr(optim):
    if optim:
        return optim.param_groups[0]['lr']
    else:
        return 0