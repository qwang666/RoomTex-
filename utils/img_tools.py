import cv2
from utils.world_points import pano_2_pers
import numpy as np


def img_cv2_inpaint(img):
    mask = img.sum(-1) == 0
    mask = mask.astype(np.uint8)
    rendered_image = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return rendered_image


def get_init_rgb_from_pano(pano_path, depth_path, scale, init_pose, f=500.0, rot=0.0):
    pano_ori_img = cv2.imread(pano_path)
    pano_ori_img = cv2.cvtColor(pano_ori_img, cv2.COLOR_BGR2RGB)
    pano_disp = 1/np.load(depth_path)

    pano_img = pano_2_pers(pano_ori_img, pano_disp,
                           init_pose, 1024*scale, f=f*scale, rot=rot)
    inpaint_pano_img = img_cv2_inpaint(pano_img)

    return inpaint_pano_img
