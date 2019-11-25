from __future__ import absolute_import
from utils.logger import logger, log
logger.setup('./logs', name='log_model002')

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

# get args from command line --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser.parse_args()


# main ------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_args()
    cfg  = Config.fromfile(args.config)
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
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    ## ------------------------------------------
    do_train(cfg, model)


# train model -----------------------------------
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
    

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')