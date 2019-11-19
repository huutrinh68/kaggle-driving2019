from __future__ import absolute_import
from utils.logger import logger, log
logger.setup('./logs', name='log_model001')

from utils.common import *
import argparse
from utils.config import Config
from utils.file import imread
import utils.kaggle as kaggle


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

    if 0:
        points_df = pd.DataFrame()
        for col in ['x', 'y', 'z', 'yaw', 'pitch', 'roll']:
            arr = []
            for ps in train['PredictionString']:
                coords = kaggle.str2coords(ps)
                arr += [c[col] for c in coords]
            points_df[col] = arr

        log.info(f'len(points_df): {len(points_df)}')
        log.info(points_df.head())
        
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

        ############
        plt.figure(figsize=(15,6))
        sns.countplot(lens)
        # plt.xlabel('Number of cars in image')
        # plt.show()
        plt.savefig('eda/number_cars_in_image.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['x'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
        # sns.distplot([kaggle.str2coords(s)[0]['x'] for s in train['PredictionString']]);
        plt.xlabel('x')
        # plt.show()
        plt.savefig('eda/x.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['y'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
        plt.xlabel('y')
        # plt.show()
        plt.savefig('eda/y.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['z'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]), bins=500)
        plt.xlabel('z')
        # plt.show()
        plt.savefig('eda/z.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['yaw'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
        plt.xlabel('yaw')
        # plt.show()
        plt.savefig('eda/yaw.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['roll'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
        plt.xlabel('roll')
        # plt.show()
        plt.savefig('eda/roll.png')

        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[c['pitch'] for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
        plt.xlabel('pitch')
        # plt.show()
        plt.savefig('eda/pitch.png')


        ############
        plt.figure(figsize=(15,6))
        sns.distplot(functools.reduce(lambda a, b: a + b, [[kaggle.rotate(c['roll'], np.pi) for c in kaggle.str2coords(s)] for s in train['PredictionString']]))
        plt.xlabel('roll rotated by pi')
        # plt.show()
        plt.savefig('eda/roll_rotated_by_pi.png')

        plt.figure(figsize=(14,14))
        plt.imshow(imread(opj(cfg.train_images, train.iloc[2217]['ImageId'] + '.jpg')))
        plt.scatter(*kaggle.get_img_coords(train.iloc[2217]['PredictionString']), color='red', s=100)
        # plt.show()
        # log.info(kaggle.get_img_coords(train.iloc[2217]['PredictionString']))

        ############
        xs, ys = [], []

        for ps in train['PredictionString']:
            x, y = kaggle.get_img_coords(ps)
            xs += list(x)
            ys += list(y)

        plt.figure(figsize=(18,18))
        plt.imshow(imread(opj(cfg.train_images, train.iloc[2217]['ImageId'] + '.jpg')), alpha=0.3)
        plt.scatter(xs, ys, color='red', s=10, alpha=0.2)
        # plt.show()
        plt.savefig('eda/xs-ys_distribution.png')

        ############
        # view distribution from the sky
        road_width = 3
        road_xs = [-road_width, road_width, road_width, -road_width, -road_width]
        road_ys = [0, 0, 500, 500, 0]

        plt.figure(figsize=(16,16))
        plt.axes().set_aspect(1)
        plt.xlim(-50,50)
        plt.ylim(0,100)

        

        # View road
        plt.fill(road_xs, road_ys, alpha=0.2, color='gray')
        plt.plot([road_width/2,road_width/2], [0,100], alpha=0.4, linewidth=4, color='white', ls='--')
        plt.plot([-road_width/2,-road_width/2], [0,100], alpha=0.4, linewidth=4, color='white', ls='--')
        
        # View cars
        # plt.scatter(points_df['x'], np.sqrt(points_df['z']**2 + points_df['y']**2), color='red', s=10, alpha=0.1)
        # plt.savefig('eda/view_from_sky.png')


        ############
        fig = px.scatter_3d(points_df, x='x', y='y', z='z',color='pitch', range_x=(-50,50), range_y=(0,50), range_z=(0,250), opacity=0.1)
        # fig.show()


        zy_slope = LinearRegression()
        X = points_df[['z']]
        y = points_df[['y']]
        zy_slope.fit(X, y)
        print('MAE without x:', mean_absolute_error(y, zy_slope.predict(X)))

        # Will use this model later
        xzy_slope = LinearRegression()
        X = points_df[['x', 'z']]
        y = points_df['y']
        xzy_slope.fit(X, y)
        print('MAE with x:', mean_absolute_error(y, xzy_slope.predict(X)))
        print('\ndy/dx = {:.3f} \ndy/dz = {:.3f}'.format(*xzy_slope.coef_))

        plt.figure(figsize=(16,16))
        plt.xlim(0,500)
        plt.ylim(0,100)
        plt.scatter(points_df['z'], points_df['y'], label='Real points')
        X_line = np.linspace(0,500, 10)
        plt.plot(X_line, zy_slope.predict(X_line.reshape(-1, 1)), color='orange', label='Regression')
        plt.legend()
        plt.xlabel('z coordinate')
        plt.ylabel('y coordinate')
        plt.savefig('eda/linear_regression.png')

        
        # 3d view
        n_rows = 6
        for idx in range(n_rows):
            fig, axes = plt.subplots(1, 2, figsize=(20,20))
            img = imread(opj(cfg.train_images, train['ImageId'].iloc[idx] + '.jpg'))
            axes[0].imshow(img)
            img_vis = kaggle.visualize(img, kaggle.str2coords(train['PredictionString'].iloc[idx]))
            axes[1].imshow(img_vis)
            # plt.show()
            plt.savefig(f'eda/img-view_coords_{idx}.png')

    
    img0 = imread(opj(cfg.train_images, train.iloc[0]['ImageId'] + '.jpg'))
    img = kaggle.preprocess_image(img0)

    mask, regr = kaggle.get_mask_and_regr(img0, train.iloc[0]['PredictionString'])
    print('img.shape', img.shape, 'std:', np.std(img))
    print('mask.shape', mask.shape, 'std:', np.std(mask))
    print('regr.shape', regr.shape, 'std:', np.std(regr))

    plt.figure(figsize=(16,16))
    plt.title('Processed image')
    plt.imshow(img)
    # plt.show()
    plt.savefig('eda/processed_image.png')

    plt.figure(figsize=(16,16))
    plt.title('Detection Mask')
    plt.imshow(mask)
    # plt.show()
    plt.savefig('eda/detection_mask.png')

    plt.figure(figsize=(16,16))
    plt.title('Yaw values')
    plt.imshow(regr[:,:,-2])
    # plt.show()
    plt.savefig('eda/yaw_values.png')



if __name__ == "__main__":
    main()