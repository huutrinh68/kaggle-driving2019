from __future__ import absolute_import
from utils.logger import logger, log
logger.setup('./outputs/logs', name='log_model_submit002')

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

model_paths = [
    '/home/citynow-cloud/data/driving2019/outputs/model002/checkpoint/0/top1.pth',
]

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
    
    make_submission(cfg)
    

def make_submission(cfg):
    predictions = []
    # setting dataset ---------------------------
    loader_test = dataset_factory.get_dataloader(cfg.data.test)
    
    ## model ------------------------------------
    model = model_factory.get_model(cfg)
    util.load_model(model_paths[0], model)
    # model.to(device)
    # model.eval()

    # for img, _, _ in tqdm(loader_test):
    #     with torch.no_grad():
    #         output = model(img.to(device))
    #     output = output.data.cpu().numpy()
    #     for out in output:
    #         coords = kaggle.extract_coords(out)
    #         s = kaggle.coords2str(coords)
    #         predictions.append(s)

    # test = pd.read_csv(cfg.data.test.dataframe)
    # test['PredictionString'] = predictions
    # test.to_csv(opj(cfg.workdir,test,'predictions.csv'), index=False)
    # log.info(test.head())


#####
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard Interrupted')