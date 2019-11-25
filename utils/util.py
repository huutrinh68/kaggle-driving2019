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

    model.load_state_dict(state['state_dict'])
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