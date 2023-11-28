from utils.file_tools import *
import glob
import numpy as np
import cv2
import torch
from utils.world_points import read_imgx4_depth_mask, img2world, world2img, get_cv_sd_mask, get_combine_img
from utils.mesh.mesh import load_mesh, load_adornment
from utils.world_points import pano_2_pers
from utils.pose_tools import get_poses
from utils.camera_tools import get_depth
import multiprocessing
from scripts.prepare_pers import trans_room_pano
import argparse
import trimesh
from utils.mesh_tools import rot_x, rot_y, rot_z
from utils.img_tools import get_init_rgb_from_pano


def load_single_obj(obj_path, total_pers_num):
    # files = glob.glob(obj_path + "/*")
    # total_pers_num = len(files)
    scale = 4
    img_list = []
    world_coords_list = []
    mask_list = []
    for i in range(total_pers_num):
        if i == 0:
            f_name = 'init'
            if os.path.exists(obj_path + '/' + f_name):
                f = np.load(obj_path + '/' + f_name + '/init_f.npy')

                img_path = obj_path + '/' + f_name + '/inpaint.png'
                depth_path = obj_path + '/' + f_name + '/depth_all_4k.npy'
                mask_path = obj_path + '/' + f_name + '/mask_obj_4k.png'
                c2w = np.load(obj_path + '/' + f_name + '/pose.npy')
                img, depth, mask, intr = read_imgx4_depth_mask(
                    img_path, depth_path, mask_path, scale, kernel_size=9, iter_num=2, f=f)
        else:
            f_name = 'pers_0{}'.format(i-1)
            if os.path.exists(obj_path + '/' + f_name):
                img_path = obj_path + '/' + f_name + '/inpaint.png'
                depth_path = obj_path + '/' + f_name + '/depth_all_4k.npy'
                mask_path = obj_path + '/' + f_name + '/mask_obj_4k.png'
                proj_mask_path = obj_path + '/' + f_name + '/proj_mask.png'
                c2w = np.load(obj_path + '/' + f_name + '/pose.npy')
                img, depth, mask, intr = read_imgx4_depth_mask(
                    img_path, depth_path, mask_path, scale, kernel_size=9, iter_num=2, init=False, sd_mask_path=proj_mask_path)
        if os.path.exists(obj_path + '/' + f_name):
            world_coords = img2world(
                img, depth, mask, intr, torch.from_numpy(c2w))
            img_list.append(img.reshape(-1, 3))
            world_coords_list.append(world_coords)
            mask_list.append(mask)

    total_img = np.concatenate(img_list, axis=0)
    total_world_coords = torch.cat(world_coords_list, dim=-1)
    total_mask = np.concatenate(mask_list, axis=-1)
    return total_img, total_world_coords, total_mask


def rot_x(theta_range):
    rotation_x = torch.tensor([
        [1.0, 0.0, 0.0, 0],
        [0.0, np.cos(theta_range), -np.sin(theta_range), 0],
        [0.0, np.sin(theta_range), np.cos(theta_range), 0],
        [0, 0, 0, 1]
    ]).float()

    return rotation_x


def rot_z(theta_range):
    rotation_z = torch.tensor([
        [np.cos(theta_range), -np.sin(theta_range), 0.0, 0],
        [np.sin(theta_range), np.cos(theta_range), 0.0, 0],
        [0.0, 0.0, 1.0, 0],
        [0, 0, 0, 1]

    ]).float()

    return rotation_z


def rot_y(phi_range):
    rotation_y = torch.tensor([
        [np.cos(phi_range), 0.0, np.sin(phi_range), 0],
        [0.0, 1.0, 0.0, 0],
        [-np.sin(phi_range), 0.0, np.cos(phi_range), 0],
        [0, 0, 0, 1]

    ]).float()

    return rotation_y


def rotate_mesh(v, ang_x, ang_y, ang_z):

    v = rot_z(ang_z*np.pi) @ rot_y(ang_y*np.pi) @ rot_x(ang_x*np.pi) @ v.T
    v = v.T

    return v


def change_axis_pts(pts):
    newpts = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    newpts[:, 0] = pts[0]
    newpts[:, 1] = -pts[2]
    newpts[:, 2] = pts[1]
    return newpts


def rotate(obj_world_coords, i):
    obj_world_coords = obj_world_coords.squeeze(0)
    dist = (change_axis_pts(cfg["obj_old_pos"][i]) -
            change_axis_pts(cfg['pano']['pano_cam_center']))
    obj_world_coords = obj_world_coords.T
    dist = dist.repeat((obj_world_coords.shape[0], 1))
    obj_world_coords = obj_world_coords-dist
    ang = np.array(cfg["obj_new_rot"][i]) - np.array(cfg["obj_old_rot"][i])
    obj_world_coords = rotate_mesh(obj_world_coords, ang[0], ang[2], ang[1])
    obj_world_coords = obj_world_coords + \
        change_axis_pts(cfg["obj_new_pos"][i]) - \
        change_axis_pts(cfg['pano']['pano_cam_center'])
    obj_world_coords = obj_world_coords.T
    obj_world_coords = obj_world_coords.unsqueeze(0)
    return obj_world_coords


def load_inpaint_pts(cfg, remove=False):
    img_list = []
    world_coords_list = []
    mask_list = []

    o_list = cfg['obj_id']
    # TODO
    # o_list = ["chair1", "chair2", "chair3", "chair4", "chair5", "desk", "table"]
    for i in range(len(o_list)):
        obj = o_list[i]

        obj_path = cfg['save_path'] + '/' + obj
        total_pers_num = len(glob.glob(obj_path + "/*"))
        obj_img, obj_world_coords, obj_mask = load_single_obj(
            obj_path, total_pers_num)

        img_list.append(obj_img)
        world_coords_list.append(obj_world_coords)
        mask_list.append(obj_mask)
    if cfg["adornment"]:
        for obj in cfg['adornment_id']:
            obj_path = cfg['save_path'] + '/' + obj
            total_pers_num = len(glob.glob(obj_path + "/*"))
            obj_img, obj_world_coords, obj_mask = load_single_obj(
                obj_path, total_pers_num)

            img_list.append(obj_img)
            world_coords_list.append(obj_world_coords)
            mask_list.append(obj_mask)
    total_img = np.concatenate(img_list, axis=0)
    total_world_coords = torch.cat(world_coords_list, dim=-1)
    total_mask = np.concatenate(mask_list, axis=-1)
    return total_img, total_world_coords, total_mask


def img_cv2_inpaint_room(img):
    mask = img.sum(-1) == 0
    mask1 = mask.astype(np.uint8)

    # mask2 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2RGB)
    mask2 = cv2.erode(mask1, np.ones((9, 9), np.uint8), iterations=1)
    mask2 = cv2.dilate(mask2, np.ones((9, 9), np.uint8), iterations=1)
    # mask2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2RGB)
    final_mask = (mask2 == 0) * (mask1 > 0)

    # rendered_image = cv2.inpaint(img, final_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    rendered_image = cv2.inpaint(
        img, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return rendered_image


def img_cv2_inpaint_obj(img, img_mask):
    mask = img.sum(-1) == 0
    mask = mask.astype(np.uint8)

    final_mask = (mask > 0) * (img_mask[:, :, 0] > 0)
    rendered_image = cv2.inpaint(
        img, final_mask.astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return rendered_image


def show_all(cfg, pose, i, save_path, scale=2, inpaint=False, remove=False):
    scale = scale
    total_img, total_world_coords, total_mask = load_inpaint_pts(cfg, remove)
    print(total_img.shape,  total_world_coords.shape, total_mask.shape)
    room_name = cfg['save_path'].split('/')[-1]

    check_path(save_path)
    # cv2.imwrite('{}/compare_{}_all_img_warp_4k.png'.format(save_path, i), cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB))
    mesh_empty_room, mesh_total_obj, mesh_obj_list, mesh_boundry = load_mesh(
        cfg)
    if cfg["adornment"]:
        mesh_adornment, mesh_adornment_list = load_adornment(cfg)
        mesh_total_obj = trimesh.util.concatenate(
            [mesh_adornment, mesh_total_obj])

    pano_center = cfg['pano']['pano_cam_center']
    mesh_total_obj = trans_room_pano(mesh_total_obj, pano_center)

    img_warp, img_warp_depth = world2img(
        pose, total_img, total_world_coords, total_mask, new_w=1024*scale, new_h=1024*scale, f=500.0*scale)

    depth = get_depth(pose, mesh_total_obj, int(1024*scale))
    mask = depth != 0.1
    mask = np.stack([mask]*3, axis=2)*255
    depth_mask = (img_warp_depth-0.02) < depth
    img_depth_warp = img_warp*depth_mask[:, :, np.newaxis]

    # cv2.imwrite('{}/img_{}_{}k_{}.png'.format(save_path, i, scale, remove), cv2.cvtColor(img_depth_warp, cv2.COLOR_BGR2RGB))
    cv2.imwrite('{}/mask_{}_{}k_{}.png'.format(save_path,
                i, scale, remove), mask)
    print("warp {} done".format(i))
    if inpaint:
        img_inpaint_depth_warp = img_cv2_inpaint_obj(img_depth_warp, mask)
        cv2.imwrite('{}/inpaint_{}_img_{}k_{}.png'.format(save_path, i, scale,
                    remove), cv2.cvtColor(img_inpaint_depth_warp, cv2.COLOR_BGR2RGB))
        print("inpaint Done")
    print("save all_{}".format(i))


def save_empty_pers(cfg, save_path, i, pose, scale, inpaint=False):
    file_path = cfg['save_path']
    # room_name = cfg['save_path'].split('/')[-1]
    pano_wall_4K_path = file_path+'/pano/image/' + 'pano_wall_4k.png'
    pano_wall_depth_4K_path = file_path+'/pano/' + 'pano_depth_wall_4k.npy'

    img = get_init_rgb_from_pano(
        pano_wall_4K_path, pano_wall_depth_4K_path, scale, pose, rot=-cfg['pano']['rot'])

    check_path(save_path)
    cv2.imwrite('{}/room_2k_{}.png'.format(save_path, i),
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print("save all_{}".format(save_path))
    if inpaint:
        inpaint_pano_img = img_cv2_inpaint_room(img)
        cv2.imwrite('{}/inpaint_room_2k_{}.png'.format(save_path, i),
                    cv2.cvtColor(inpaint_pano_img, cv2.COLOR_BGR2RGB))


if __name__ == "__main__":
    cfg = load_cfg('demo/configs/livingroom.yaml')
    save_path = "demo/results/livingroom"
    position = torch.tensor([[[-1.2, 0.5, 0.4]],

                             ])
    center = torch.tensor([[[1.5, -1.1, -0.2]],

                           ])

    for i in range(len(position)):
        print(i)
        pose = get_poses(position[i], center[i])
        show_all(cfg, pose, i, save_path, 2, True)
        save_empty_pers(cfg, save_path, i, pose, 2, inpaint=True)
