from __future__ import absolute_import
from utils.logger import logger, log
logger.setup('./outputs/logs', name='log_model003')

from utils.common import *
import argparse
from utils.config import Config
from utils.file import imread
import utils.kaggle as kaggle
import utils.util as util
import utils.file as file

from datasets import dataset_factory
from models import model_factory
from optimizers import optimizer_factory
from schedulers import scheduler_factory
from losses import criterion_factory

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3'
torch.backends.cudnn.benchmark=True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### get args from command line ---------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('--fold', nargs='+', type=int)
    return parser.parse_args()


###### main -------------------------------------
def main():
    args = get_args()
    cfg  = Config.fromfile(args.config)
    cfg.fold = args.fold
    global device
    cfg.device = device
    log.info(cfg)

    # torch.cuda.set_device(cfg.gpu)
    util.set_seed(cfg.seed)
    log.info(f'setting seed = {cfg.seed}')
    
    # setup -------------------------------------
    for f in ['checkpoint', 'train', 'valid', 'test', 'backup']: os.makedirs(cfg.workdir+'/'+f, exist_ok=True)
    if 0: #not work perfect
        file.backup_project_as_zip(PROJECT_PATH, cfg.workdir+'/backup/code.train.%s.zip'%IDENTIFIER)

    ## model ------------------------------------
    model = model_factory.get_model(cfg)

    # multi-gpu----------------------------------
    if torch.cuda.device_count() > 1 and len(cfg.gpu) > 1:
        log.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    model.to(device)

    ## train model-------------------------------
    do_train(cfg, model)



###### train model ------------------------------
def do_train(cfg, model):
    # get criterion -----------------------------
    criterion = criterion_factory.get_criterion(cfg)
    
    # get optimization --------------------------
    optimizer = optimizer_factory.get_optimizer(model, cfg)

    # initial -----------------------------------
    best = {
        'loss': float('inf'),
        'score': 0.0,
        'epoch': -1,
    }

    # resume model ------------------------------
    if cfg.resume_from:
        log.info('\n')
        log.info(f're-load model from {cfg.resume_from}')
        detail = util.load_model(cfg.resume_from, model, optimizer, cfg.device)
        best.update({
            'loss': detail['loss'],
            'score': detail['score'],
            'epoch': detail['epoch'],
        })

    # scheduler ---------------------------------
    scheduler = scheduler_factory.get_scheduler(cfg, optimizer, best['epoch'])
    
    # fp16 --------------------------------------
    if cfg.apex:
        amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # setting dataset ---------------------------
    loader_train = dataset_factory.get_dataloader(cfg.data.train)
    loader_valid = dataset_factory.get_dataloader(cfg.data.valid)

    # start trainging ---------------------------
    start_time = datetime.now().strftime('%Y/%m/%d %H:%M:%S')
    log.info('\n')
    log.info(f'** start train [fold{cfg.fold}th] {start_time} **\n')
    log.info('epoch    iter      rate     | smooth_loss/score | valid_loss/score | best_epoch/best_score |  min')
    log.info('-------------------------------------------------------------------------------------------------')

    for epoch in range(best['epoch']+1, cfg.epoch):
        end = time.time()
        util.set_seed(epoch)

        ## train model --------------------------
        train_results = run_nn(cfg.data.train, 'train', model, loader_train, criterion=criterion, optimizer=optimizer, apex=cfg.apex, epoch=epoch)
    
        ## valid model --------------------------
        with torch.no_grad():
            val_results = run_nn(cfg.data.valid, 'valid', model, loader_valid, criterion=criterion, epoch=epoch)
        
        detail = {
            'score': val_results['score'],
            'loss': val_results['loss'],
            'epoch': epoch,
        }

        if val_results['loss'] <= best['loss']:
            best.update(detail)
            util.save_model(model, optimizer, detail, cfg.fold[0], os.path.join(cfg.workdir, 'checkpoint'))


        log.info('%5.1f   %5d    %0.6f   |  %0.4f  %0.4f  %0.4f  %0.4f  |  %0.4f  %0.4f  %0.4f  %6.4f |  %6.1f     %6.4f    | %3.1f min' % \
            (epoch+1, len(loader_train), util.get_lr(optimizer), train_results['loss'], train_results['mask_loss'], train_results['regr_loss'], train_results['score'], \
                val_results['loss'], val_results['mask_loss'], val_results['regr_loss'], val_results['score'], \
                    best['epoch'], best['score'], (time.time() - end) / 60))
        
        if cfg.scheduler.name == 'StepLR':
            scheduler.step()
        elif cfg.scheduler.name == 'ReduceLROnPlateau':
            scheduler.step(val_results['loss'])

        # early stopping-------------------------
        if cfg.early_stop:
            if epoch - best['epoch'] > cfg.early_stop:
                log.info(f'=================================> early stopping!')
                break
        time.sleep(0.01)

##### run cnn------------------------------------
def run_nn(
    cfg,
    mode,
    model, 
    loader, 
    criterion=None, 
    optimizer=None, 
    scheduler=None,
    apex=None, 
    epoch=None):

    if mode in ['train']:
        model.train()
    elif mode in ['valid', ]:
        model.eval()
    else:
        raise 

    losses = util.AverageMeter()
    mask_losses = util.AverageMeter()
    regr_losses = util.AverageMeter()
    scores = util.AverageMeter()

    ids_all = []
    targets_all = []
    outputs_all = []

    # log.info(f'len(loader): {len(loader)}')
    for i, (inputs, targets, regrs) in enumerate(loader):
        # log.info(f'i: {i}')
        # zero out gradients so we can accumulate new ones over batches
        if mode in ['train']:
            optimizer.zero_grad()

        # move data to device
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        regrs = regrs.to(device, dtype=torch.float)

        # log.info(f'inputs.shape: {inputs.shape}')
        # log.info(f'targets.shape: {targets.shape}')
        # log.info(f'regrs.shape: {regrs.shape}')

        outputs = model(inputs)
        # log.info(f'outputs.shape: {outputs.shape}')

        # both train mode and valid mode
        if mode in ['train', 'valid']:
            with torch.set_grad_enabled(mode == 'train'):
                if epoch < cfg.switch_loss_epoch:
                    loss, mask_loss, regr_loss = criterion(outputs, targets, regrs, 1)
                else:
                    loss, mask_loss, regr_loss = criterion(outputs, targets, regrs, 0.5)

                loss = loss/cfg.n_grad_acc
                mask_loss = mask_loss/cfg.n_grad_acc
                regr_loss = regr_loss/cfg.n_grad_acc
                with torch.no_grad():
                    losses.update(loss.item())
                    mask_loss.update(mask_loss.item())
                    regr_loss.update(regr_loss.item())
        
        # train mode
        if mode in ['train']:
            if apex:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward() # accumulate loss
            
            if (i+1) % cfg.n_grad_acc == 0 or (i+1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step() # update
                optimizer.zero_grad() # flush

        # # compute metrics
        # score = metrics.dice_score(outputs, targets)
        # with torch.no_grad():
        #     scores.update(score.item())

        
    result = {
        'loss': losses.avg,
        'mask_loss': mask_loss.avg,
        'regr_loss': regr_loss.avg,
        'score': scores.avg,
        'ids': ids_all,
        'targets': np.array(targets_all),
        'outputs': np.array(outputs_all),
    }

    return result



#####
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')