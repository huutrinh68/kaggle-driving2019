from utils.logger import logger, log
from utils.common import *
from utils.file import imread
from utils import kaggle as kaggle

class CarDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg 
        self.df = pd.read_csv(self.cfg.dataframe)
        if cfg.mode in ['train', 'valid']:
            self.df_train, self.df_valid = self.train_valid_split(self.df)
            if cfg.mode == 'train':
                self.df = self.df_train
            elif cfg.mode == 'valid':
                self.df = self.df_valid

        print(f'mode: {cfg.mode}, len(df): {len(self.df)}')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image_name, labels = self.df.values[idx]
        # print(f'image_name: {image_name}')
        # print(f'labels: {labels}')
        image_path = opj(self.cfg.img_dir, image_name+'.jpg')

        flip = False
        if self.cfg.mode == 'train':
            flip = np.random.rand() > 0.5

        origin_image = imread(image_path, fast_mode=True)
        img = kaggle.preprocess_image(origin_image, flip=flip)
        img = np.rollaxis(img, 2, 0)

        if self.cfg.mode in ['train', 'valid']:
            mask, regr = kaggle.get_mask_and_regr(origin_image, labels, flip=flip)
            regr = np.rollaxis(regr, 2, 0)
        else:
            mask, regr = 0, 0
        # print(f'mask: {mask}')
        # print(f'regr: {regr}')

        return [img, mask, regr]

    
    def train_valid_split(self, df):
        df_train, df_valid = train_test_split(df, test_size=0.01, random_state=42)
        return df_train, df_valid
    

# dataloader
def get_dataloader(cfg):
    dataset = CarDataset(cfg)
    loader = DataLoader(dataset, **cfg.loader)

    return loader










