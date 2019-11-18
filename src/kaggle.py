import numpy as np 
from math import sin, cos
import cv2


img_width = 1024
img_height = (img_width//16)*5
model_scale = 8

camera_matrix = np.array([
    [2304.5479, 0,  1686.2379],
    [0, 2305.8757, 1354.9849],
    [0, 0, 1]], dtype=np.float32)

def str2coords(s, names=['id', 'yaw', 'pitch', 'roll', 'x', 'y', 'z']):
    coords = []
    for l in np.array(s.split()).reshape([-1, 7]):
        coords.append(dict(zip(names, l.astype('float'))))
        if 'id' in coords[-1]:
            coords[-1]['id'] = int(coords[-1]['id'])
    return coords

def rotate(x, angle):
    x = x + angle
    x = x - (x + np.pi) // (2 * np.pi) * 2 * np.pi
    return x

def get_img_coords(s):
    coords = str2coords(s)
    xs = [c['x'] for c in coords]
    ys = [c['y'] for c in coords]
    zs = [c['z'] for c in coords]
    P = np.array(list(zip(xs, ys, zs))).T
    img_p = np.dot(camera_matrix, P).T
    img_p[:, 0] /= img_p[:, 2]
    img_p[:, 1] /= img_p[:, 2]
    img_xs = img_p[:, 0]
    img_ys = img_p[:, 1]
    img_zs = img_p[:, 2] # z = Distance from the camera
    return img_xs, img_ys


# convert euler angle to rotation matrix
# inspect here?
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
def euler_to_rot(yaw, pitch, roll):
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
    
    # print(f'case1: \n{np.dot(Y, np.dot(P, R))}')
    # print('\n')
    # print(f'case2: \n{np.dot(R, np.dot(Y, P))}')
    # return np.dot(Y, np.dot(P, R))
    return np.dot(R, np.dot(Y, P))


def draw_line(image, points):
    color = (255, 0, 0)
    cv2.line(image, tuple(points[0][:2]), tuple(points[3][:2]), color, 16)
    cv2.line(image, tuple(points[0][:2]), tuple(points[1][:2]), color, 16)
    cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
    cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
    return image


# def draw_line(image, points):
#     color = (255, 0, 0)
#     cv2.line(image, tuple(points[1][:2]), tuple(points[2][:2]), color, 16)
#     cv2.line(image, tuple(points[1][:2]), tuple(points[4][:2]), color, 16)

#     cv2.line(image, tuple(points[1][:2]), tuple(points[5][:2]), color, 16)
#     cv2.line(image, tuple(points[2][:2]), tuple(points[3][:2]), color, 16)
#     cv2.line(image, tuple(points[2][:2]), tuple(points[6][:2]), color, 16)
#     cv2.line(image, tuple(points[3][:2]), tuple(points[4][:2]), color, 16)
#     cv2.line(image, tuple(points[3][:2]), tuple(points[7][:2]), color, 16)

#     cv2.line(image, tuple(points[4][:2]), tuple(points[8][:2]), color, 16)
#     cv2.line(image, tuple(points[5][:2]), tuple(points[8][:2]), color, 16)

#     cv2.line(image, tuple(points[5][:2]), tuple(points[6][:2]), color, 16)
#     cv2.line(image, tuple(points[6][:2]), tuple(points[7][:2]), color, 16)
#     cv2.line(image, tuple(points[7][:2]), tuple(points[8][:2]), color, 16)
#     return image


def draw_points(image, points):
    for (p_x, p_y, p_z) in points:
        cv2.circle(image, (p_x, p_y), int(1000 / p_z), (0, 255, 0), -1)
#         if p_x > image.shape[1] or p_y > image.shape[0]:
#             print('Point', p_x, p_y, 'is out of image with shape', image.shape)
    return image

# def draw_points(image, points):
#     image = np.array(image)
#     for (p_x, p_y, p_z) in points:
#         print(f'p_x, p_y: {p_x}, {p_y}')
#         cv2.circle(image, (p_x, p_y), 5, (255, 0, 0), -1)
#     return image


def visualize(img, coords):
    # You will also need functions from the previous cells
    x_l = 1.02
    y_l = 0.80
    z_l = 2.31
    
    img = img.copy()
    for point in coords:
        # Get values
        x, y, z = point['x'], point['y'], point['z']
        yaw, pitch, roll = -point['pitch'], -point['yaw'], -point['roll']
        # Math
        Rt = np.eye(4)
        t = np.array([x, y, z])
        Rt[:3, 3] = t
        Rt[:3, :3] = euler_to_rot(yaw, pitch, roll).T
        Rt = Rt[:3, :]
        P = np.array([[x_l, -y_l, -z_l, 1],
                      [x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, z_l, 1],
                      [-x_l, -y_l, -z_l, 1],
                      [0, 0, 0, 1]]).T
        img_cor_points = np.dot(camera_matrix, np.dot(Rt, P))
        img_cor_points = img_cor_points.T
        img_cor_points[:, 0] /= img_cor_points[:, 2]
        img_cor_points[:, 1] /= img_cor_points[:, 2]
        img_cor_points = img_cor_points.astype(int)

        # Drawing
        img = draw_line(img, img_cor_points)
        img = draw_points(img, img_cor_points[-1:])
    
    return img


def _regr_preprocess(regr_dict, flip=False):
    if flip:
        for k in ['x', 'pitch', 'roll']:
            regr_dict[k] = -regr_dict[k]
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] / 100

    regr_dict['roll'] = rotate(regr_dict['roll'], np.pi)
    regr_dict['pitch_sin'] = sin(regr_dict['pitch'])
    regr_dict['pitch_cos'] = cos(regr_dict['pitch'])
    regr_dict.pop('pitch')
    regr_dict.pop('id')
    return regr_dict


def _regr_back(regr_dict):
    for name in ['x', 'y', 'z']:
        regr_dict[name] = regr_dict[name] * 100
    regr_dict['roll'] = rotate(regr_dict['roll'], -np.pi)
    
    pitch_sin = regr_dict['pitch_sin'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    pitch_cos = regr_dict['pitch_cos'] / np.sqrt(regr_dict['pitch_sin']**2 + regr_dict['pitch_cos']**2)
    regr_dict['pitch'] = np.arccos(pitch_cos) * np.sign(pitch_sin)
    return regr_dict


def preprocess_image(img, flip=False):
    img = img[img.shape[0] // 2:]
    bg = np.ones_like(img) * img.mean(1, keepdims=True).astype(img.dtype)
    bg = bg[:, :img.shape[1] // 6]
    img = np.concatenate([bg, img, bg], 1)
    img = cv2.resize(img, (img_width, img_height))
    if flip:
        img = img[:,::-1]
    return (img / 255).astype('float32')


def get_mask_and_regr(img, labels, flip=False):
    mask = np.zeros([img_height // model_scale, img_width // model_scale], dtype='float32')
    regr_names = ['x', 'y', 'z', 'yaw', 'pitch', 'roll']
    regr = np.zeros([img_height // model_scale, img_width // model_scale, 7], dtype='float32')
    coords = str2coords(labels)
    xs, ys = get_img_coords(labels)
    for x, y, regr_dict in zip(xs, ys, coords):
        x, y = y, x
        x = (x - img.shape[0] // 2) * img_height / (img.shape[0] // 2) / model_scale
        x = np.round(x).astype('int')
        y = (y + img.shape[1] // 6) * img_width / (img.shape[1] * 4/3) / model_scale
        y = np.round(y).astype('int')
        if x >= 0 and x < img_height // model_scale and y >= 0 and y < img_width // model_scale:
            mask[x, y] = 1
            regr_dict = _regr_preprocess(regr_dict, flip)
            regr[x, y] = [regr_dict[n] for n in sorted(regr_dict)]
    if flip:
        mask = np.array(mask[:,::-1])
        regr = np.array(regr[:,::-1])
    return mask, regr