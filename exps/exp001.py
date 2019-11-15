from __future__ import absolute_import
from src.utils.logger import logger, log
logger.setup('./logs', name='log_model001')

from src.utils.common import *
import argparse
from src.utils.config import Config
from src.utils.file import imread

import src.kaggle as kaggle


# get args from command line --------------------
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    return parser.parse_args()


# main
def main():
    args = get_args()
    cfg  = Config.fromfile(args.config)


    train = pd.read_csv(cfg.train_csv)
    camera_matrix_inv = np.linalg.inv(kaggle.camera_matrix)

    img = imread(opj(cfg.train_images, train.iloc[0]['ImageId']+'.jpg'))
    # plt.figure(figsize=(15,8))
    # plt.imshow(img)
    # plt.show()

    # log.info(train.head())
    # log.info(kaggle.camera_matrix)
    pred_string = train.iloc[0]['PredictionString']
    coords = kaggle.str2coords(pred_string)
    # log.info(coords)

    lens = [len(kaggle.str2coords(s)) for s in train['PredictionString']]

    plt.figure(figsize=(15,6))
    sns.countplot(lens)
    # plt.xlabel('Number of cars in image')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['x'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
    # sns.distplot([kaggle.str2coords(s)[0]['x'] for s in train['PredictionString']]);
    plt.xlabel('x')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['y'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
    plt.xlabel('y')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['z'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
    plt.xlabel('z')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['yaw'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
    plt.xlabel('yaw')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['roll'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
    plt.xlabel('roll')
    # plt.show()

    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[c['pitch'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
    plt.xlabel('pitch')
    # plt.show()



    plt.figure(figsize=(15,6))
    sns.distplot(functools.reduce(lambda a, b: a + b, [[kaggle.rotate(c['roll'], np.pi) for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
    plt.xlabel('roll rotated by pi')
    # plt.show()

    plt.figure(figsize=(14,14))
    plt.imshow(imread(opj(cfg.train_images, train.iloc[2217]['ImageId'] + '.jpg')))
    plt.scatter(*kaggle.get_img_coords(train.iloc[2217]['PredictionString']), color='red', s=100)
    # plt.show()
    # log.info(kaggle.get_img_coords(train.iloc[2217]['PredictionString']))

if __name__ == "__main__":
    main()