from torch.optim import lr_scheduler
from utils.logger import log

def get_scheduler(cfg, optimizer, last_epoch):
    if cfg.scheduler.name == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            **cfg.scheduler.params,
        )
        scheduler.last_epoch = last_epoch
    elif cfg.scheduler.name == 'StepLR':
        scheduler = lr_scheduler.StepLR(optimizer, **cfg.scheduler.params)
    else:
        scheduler = getattr(lr_scheduler, cfg.scheduler.name)(
            optimizer,
            last_epoch=last_epoch,
            **cfg.scheduler.params,
        )
    log.info('\n')
    log.info('** scheduler setting **')
    log.info(f'scheduler: {cfg.scheduler}')
    return scheduler