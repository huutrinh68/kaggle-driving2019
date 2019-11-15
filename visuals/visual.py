#https://www.kaggle.com/zstusnoopy/visualize-the-location-and-3d-bounding-box-of-car

import numpy as np 
import pandas as pd 
from math import sin, cos

import os 

# for dirname, _, filenames in os.walk('./'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

import matplotlib.pylab as plt 
from PIL import ImageDraw, Image
import cv2

def main():
    train = pd.read_csv('./data/train.csv')
    # camera instrinsic matrix get from camera_instrinsic.text
    # https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/agartia3/index.html figure2
    k = np.array([
        [2304.5479, 0,  1686.2379],
        [0, 2305.8757, 1354.9849],
        [0, 0, 1]], 
        dtype=np.float32)

    print(f'train header: \n{train.head()}')
    print(f'instrinsic matrix: \n{k}')

    plt.rcParams["axes.grid"] = False
    image_id = train.iloc[10]['ImageId']
    pred_string = train.iloc[10]['PredictionString']
    fig, ax = plt.subplots(figsize=(10, 10))
    img = Image.open('./data/train_images/' + image_id + '.jpg')
    # img = np.asarray(img)
    # plt.imshow(img)
    # plt.show()

    items = pred_string.split(' ')
    model_types, yaws, pitches, rolls, xs, ys, zs = [items[i::7] for i in range(7)]
    print(f'model_types: \n{model_types}')
    print(f'yaws: \n{yaws}')
    print(f'pitches: \n{pitches}')
    print(f'rolls: \n{rolls}')


    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    for yaw, pitch, roll, x, y, z in zip(yaws, pitches, rolls, xs, ys, zs):
        yaw, pitch, roll, x, y, z = [float(x) for x in [yaw, pitch, roll, x, y, z]]
        # I think the pitch and yaw should be exchanged
        yaw, pitch, roll = -pitch, -yaw, -roll
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_Rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[0, 0, 0, 1],
                    [x_l, y_l, -z_l, 1],
                    [x_l, y_l, z_l, 1],
                    [-x_l, y_l, z_l, 1],
                    [-x_l, y_l, -z_l, 1],
                    [x_l, -y_l, -z_l, 1],
                    [x_l, -y_l, z_l, 1],
                    [-x_l, -y_l, z_l, 1],
                    [-x_l, -y_l, -z_l, 1]]).T
        img_cor_points = np.dot(k, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        # call this function before chage the dtype
        img_cor_2_world_cor(img_cor_points, k)
        img_cor_points = img_cor_points.astype(int)
        img = draw_points(img, img_cor_points)
        img = draw_line(img, img_cor_points)
        
    img = Image.fromarray(img)
    plt.imshow(img)
    plt.show()


# convert euler angle to rotation matrix
# inspect here?
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def euler_to_Rot(yaw, pitch, roll):
    # Ry
    Y = np.array([[cos(yaw), 0, sin(yaw)],
                  [0, 1, 0],
                  [-sin(yaw), 0, cos(yaw)]])
    
    # Rx
    P = np.array([[1, 0, 0],
                  [0, cos(pitch), -sin(pitch)],
                  [0, sin(pitch), cos(pitch)]])
    
    # Rz
    R = np.array([[cos(roll), -sin(roll), 0],
                  [sin(roll), cos(roll), 0],
                  [0, 0, 1]])
    
    print(f'case1: \n{np.dot(Y, np.dot(P, R))}')
    print('\n')
    print(f'case2: \n{np.dot(R, np.dot(Y, P))}')
    # return np.dot(Y, np.dot(P, R))
    return np.dot(R, np.dot(Y, P))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 16)

    cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 16)
    cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 16)
    cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 16)

    cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 16)
    cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 16)

    cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 16)
    cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 16)
    cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 16)
    return image


def draw_points(image, points):
    image = np.array(image)
    for (p_x, p_y, p_z) in points:
        print(f'p_x, p_y: {p_x}, {p_y}')
        cv2.circle(image, (p_x, p_y), 5, (255, 0, 0), -1)
    return image


# image coordinate to world coordinate
# https://mem-archive.com/2018/10/28/post-962/
# https://mem-archive.com/2018/10/13/post-682/
# http://www.cse.psu.edu/~rtc12/CSE486/lecture12.pdf
def img_cor_2_world_cor(img_cor_points, k):
    x_img, y_img, z_img = img_cor_points[0]
    xc, yc, zc = x_img*z_img, y_img*z_img, z_img
    p_cam = np.array([xc, yc, zc])
    xw, yw, zw = np.dot(np.linalg.inv(k), p_cam)
    print(xw, yw, zw)
    # print(x, y, z)



if __name__ == "__main__":
    main()
    # euler_to_Rot(20, 180, 0)

