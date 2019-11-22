from utils.logger import logger, log
from utils.common import *
from utils.file import imread
from utils import kaggle as kaggle

class CarDataset(Dataset):

    def __init__(self, cfg):
        self.cfg = cfg 
        self.df = pd.read_csv(self.cfg.train_csv)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_name, labels = self.df.values[idx]
        print(image_name)
        print(labels)
        image_path = opj(self.cfg.train_images, image_name+'.jpg')

        flip = False
        if self.cfg.train_mode:
            flip = np.random.rand() > 0.5

        origin_image = imread(image_path, fast_mode=True)
        img = kaggle.preprocess_image(origin_image, flip=flip)
        img = np.rollaxis(img, 2, 0)

        mask, regr = kaggle.get_mask_and_regr(origin_image, labels, flip=flip)

        return [img, mask, regr]
    

# dataloader
def get_dataloader(cfg):
    dataset = CarDataset(cfg)
    loader = DataLoader(dataset, **cfg.loader)

    return loader










