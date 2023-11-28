from utils.camera_tools import capture_pano_depth
from utils.mesh.mesh import load_mesh, load_adornment
import os
import yaml
from utils.file_tools import check_path, load_cfg
import trimesh
import cv2
import numpy as np
import argparse


def prepare_pano_depth(cfg):

    pano_cam_center = cfg['pano']['pano_cam_center']
    rot = cfg['pano']['rot']
    save_file = cfg['save_path'] + "/pano/"
    check_path(save_file)
    mesh_empty_room, mesh_total_obj, mesh_obj_list, mesh_boundry = load_mesh(
        cfg)
    if cfg["adornment"]:
        mesh_adornment, mesh_adornment_list = load_adornment(cfg)
        mesh_total_obj = trimesh.util.concatenate(
            [mesh_adornment, mesh_total_obj])
    mesh_all = trimesh.util.concatenate([mesh_empty_room, mesh_total_obj])
    mesh_boundry_wall = trimesh.util.concatenate(
        [mesh_boundry, mesh_empty_room])
    # 2K image
    img_h = 1024
    depthmap, disp, mask = capture_pano_depth(
        mesh_empty_room, img_h, rot, pano_cam_center)
    cv2.imwrite(save_file+"/pano_wall_disp_2k.png",
                ((disp-disp.min())/(disp.max()-disp.min()))*255)
    np.save(save_file+'/pano_wall_disp_2k',
            ((disp-disp.min())/(disp.max()-disp.min()))*255)

    depthmap, disp, mask = capture_pano_depth(
        mesh_total_obj, img_h, rot, pano_cam_center)
    cv2.imwrite(save_file+'/pano_mask_obj_2k.png', mask.astype(np.uint8))

    depthmap, disp, mask = capture_pano_depth(
        mesh_all, img_h, rot, pano_cam_center)
    cv2.imwrite(save_file+"/pano_all_disp_2k.png",
                ((disp-disp.min())/(disp.max()-disp.min()))*255)
    np.save(save_file+'/pano_all_disp_2k',
            ((disp-disp.min())/(disp.max()-disp.min()))*255)

    # 4K image
    img_h = 2048
    depthmap, disp, mask = capture_pano_depth(
        mesh_all, img_h, rot, pano_cam_center)
    np.save(save_file+'/pano_depth_all_4k', depthmap)

    depthmap, disp, mask = capture_pano_depth(
        mesh_boundry_wall, img_h, rot, pano_cam_center)
    np.save(save_file+'/pano_depth_wall_4k', depthmap)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    prepare_pano_depth(cfg)
