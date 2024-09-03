import torch
import trimesh
import cv2
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
import glob
from utils.mesh.mesh import load_mesh, load_adornment
from tqdm import tqdm
from utils.mesh_tools import get_mesh_bound, get_mesh_center
import multiprocessing
from utils.pose_tools import get_poses, get_four_up_down_pose, get_focus_center, rot_error, get_four_ador_pose
from utils.file_tools import check_path, load_cfg
from utils.img_tools import get_init_rgb_from_pano
from utils.camera_tools import get_depth
import argparse


def trans_room_pano(mesh, pano_center):
    v, f = mesh.vertices, mesh.faces
    v[:, 0] -= pano_center[0]
    v[:, 1] -= pano_center[1]
    v[:, 2] -= pano_center[2]

    new_v = v.copy()
    new_v[:, 1] = -v[:, 2]
    new_v[:, 2] = v[:, 1]
    mesh = trimesh.Trimesh(vertices=new_v, faces=f)
    return mesh


def adaptable_f(c2w_ori, mesh_obj):
    f = 500.0
    depth = get_depth(c2w_ori, mesh_obj, 1024, f=f)
    mask = depth != 0.1
    h_d = np.where(mask)[0].max()-np.where(mask)[0].min()
    w_d = np.where(mask)[1].max()-np.where(mask)[1].min()
    while (w_d < 512):
        f += 100.0
        depth = get_depth(c2w_ori, mesh_obj, 1024, f=f)
        mask = depth != 0.1
        h_d = np.where(mask)[0].max()-np.where(mask)[0].min()
        w_d = np.where(mask)[1].max()-np.where(mask)[1].min()
    while (h_d > 1024*0.95 or w_d > 1024*0.95):
        f -= 100.0
        depth = get_depth(c2w_ori, mesh_obj, 1024, f=f)
        mask = depth != 0.1
        h_d = np.where(mask)[0].max()-np.where(mask)[0].min()
        w_d = np.where(mask)[1].max()-np.where(mask)[1].min()
    return f


def save_init_depth(c2w_ori, mesh_obj, room_obj, mesh_total_obj, file_path, pano_all_4K_path, pano_all_depth_4K_path, rot=0.0):
    save_path = file_path + '/init/'
    check_path(save_path)

    f = adaptable_f(c2w_ori, mesh_obj)
    print(f)
    # save depth npy & disp png
    depth = get_depth(c2w_ori, room_obj, 1024, f=f)
    np.save(save_path + "/depth_all", depth)

    disp = 1/depth
    disp[disp == 10] = disp.min()
    cv2.imwrite(save_path + '/disp_all.png',
                (disp-disp.min())/(disp.max()-disp.min())*255)

    depth = get_depth(c2w_ori, room_obj, 4096, f=f)

    np.save(save_path + '/depth_all_4k', depth)
    # save mask png
    depth = get_depth(c2w_ori, mesh_obj, 1024, f=f)
    mask = depth != 0.1
    mask = np.stack([mask]*3, axis=2)*255
    cv2.imwrite(save_path + '/mask_obj.png', mask.astype(np.uint8))
    sd_mask = cv2.dilate(mask.astype(np.uint8), np.ones(
        (15, 15), np.uint8), iterations=2)
    cv2.imwrite(save_path + '/sd_mask.png', sd_mask)

    occ_depth = get_depth(c2w_ori, mesh_total_obj, 1024, f=f)
    occ_disp = 1/occ_depth
    occ_disp[occ_disp == 10] = occ_disp.min()
    cv2.imwrite(save_path + '/disp_occ.png', (occ_disp -
                occ_disp.min())/(occ_disp.max()-occ_disp.min())*255)
    occ_mask = occ_depth+0.001 < depth
    occ_mask = cv2.dilate(occ_mask.astype(np.uint8),
                          np.ones((7, 7), np.uint8), iterations=2)
    occ_mask = np.stack([occ_mask]*3, axis=2)*255
    cv2.imwrite(save_path + '/occ_mask.png', occ_mask)

    depth = get_depth(c2w_ori, mesh_obj, 4096, f=f)
    mask = depth != 0.1
    mask = np.stack([mask]*3, axis=2)*255
    cv2.imwrite(save_path + '/mask_obj_4k.png', mask.astype(np.uint8))

    # save pose
    np.save(save_path + '/pose', c2w_ori.numpy())
    np.save(save_path + '/init_f', f)
    # save initial pano2pers img
    # init_img = get_init_rgb_from_pano(pano_all_4K_path, pano_all_depth_4K_path, c2w_ori, f=f, rot=rot)
    # cv2.imwrite(save_path + '/img_init.png', cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB))


def save_pers_img(c2w_four, mesh_obj, room_obj, pano_wall_4K_path, pano_wall_depth_4K_path, save_path, rot=0.0):
    # if os.path.exists(save_path):
    #     pass
    # else:
    check_path(save_path)

    # save depth npy & disp png
    depth = get_depth(c2w_four, room_obj, 1024, f=500.0)
    np.save(save_path + "/depth_all", depth)

    disp = 1/depth
    disp[disp == 10] = disp.min()
    cv2.imwrite(save_path + '/disp_all.png',
                (disp-disp.min())/(disp.max()-disp.min())*255)

    depth = get_depth(c2w_four, room_obj, 4096, f=500.0)
    np.save(save_path + "/depth_all_4k", depth)
    # save mask png
    depth = get_depth(c2w_four, mesh_obj, 1024, f=500.0)
    mask = depth != 0.1
    mask = np.stack([mask]*3, axis=2)*255
    cv2.imwrite(save_path + '/mask_obj.png', mask.astype(np.uint8))

    depth = get_depth(c2w_four, mesh_obj, 4096, f=500.0)
    mask = depth != 0.1
    mask = np.stack([mask]*3, axis=2)*255
    cv2.imwrite(save_path + '/mask_obj_4k.png', mask.astype(np.uint8))

    # save pose
    np.save(save_path + "/pose", c2w_four.numpy())

    # save empty room img
    # img = get_init_rgb_from_pano(pano_wall_4K_path, pano_wall_depth_4K_path, c2w_four, rot=rot)
    # cv2.imwrite(save_path + "/room_empty.png", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pre_obj_depth(mesh_obj, mesh_room, mesh_total_obj, room_bound, file_path, obj_file_path, rot=0.0):
    obj_center = get_mesh_center(mesh_obj)
    position = torch.tensor([[0.0, 0.0, 0.0]])
    c2w_ori = get_poses(position, obj_center)
    c2w_four, c2w_down = get_four_up_down_pose(mesh_obj, room_bound, scale=0.7)
    c2w_focus = get_focus_center(mesh_obj, room_bound, scale=0.9)
    room_obj = trimesh.util.concatenate([mesh_room, mesh_obj])

    pano_all_4K_path = file_path+'/pano/image/' + 'pano_all_4k.png'
    pano_all_depth_4K_path = file_path+'/pano/' + 'pano_depth_all_4k.npy'
    pano_wall_4K_path = file_path+'/pano/image/' + 'pano_wall_4k.png'
    pano_wall_depth_4K_path = file_path+'/pano/' + 'pano_depth_wall_4k.npy'

    save_init_depth(c2w_ori, mesh_obj, room_obj, mesh_total_obj,
                    obj_file_path, pano_all_4K_path, pano_all_depth_4K_path, rot=rot)

    # choose angle distance smallest fist
    angle_distance_list = []
    new_c2w_four = []

    for i in range(len(c2w_four)):
        angle_distance = rot_error(
            c2w_ori[0, :3, :3].numpy(), c2w_four[i, :3, :3].numpy())
        angle_distance_list.append(angle_distance)
    sorted_angle_id = sorted(range(len(angle_distance_list)),
                             key=lambda k: angle_distance_list[k], reverse=False)
    for i in range(len(sorted_angle_id)):
        if angle_distance_list[sorted_angle_id[i]] > 15.0:
            new_c2w_four.append(c2w_four[sorted_angle_id[i]])
    new_c2w_four = torch.stack(new_c2w_four)

    angle_distance_list = []
    new_c2w_down = []
    for i in range(len(c2w_down)):
        angle_distance = rot_error(
            c2w_ori[0, :3, :3].numpy(), c2w_down[i, :3, :3].numpy())
        angle_distance_list.append(angle_distance)
    sorted_angle_id = sorted(range(len(angle_distance_list)),
                             key=lambda k: angle_distance_list[k], reverse=False)
    for i in range(len(sorted_angle_id)):
        new_c2w_down.append(c2w_down[sorted_angle_id[i]])
    new_c2w_down = torch.stack(new_c2w_down)
    if len(c2w_focus) > 0:
        c2w_four = torch.concat([new_c2w_four, new_c2w_down, c2w_focus])
    else:
        c2w_four = torch.concat([new_c2w_four, new_c2w_down])

    for i in tqdm(range(len(c2w_four))):
        save_path = obj_file_path + '/pers_0{}/'.format(i)
        save_pers_img(c2w_four[i].unsqueeze(0), mesh_obj, room_obj,
                      pano_wall_4K_path, pano_wall_depth_4K_path, save_path, rot=rot)


def pre_ador_depth(mesh_obj, mesh_room, mesh_total_obj, room_bound, file_path, obj_file_path, rot=0.0):
    room_obj = trimesh.util.concatenate([mesh_room, mesh_obj])
    obj_center = get_mesh_center(mesh_obj)
    position = torch.tensor([[0.0, 0.0, 0.0]])
    c2w_ori = get_poses(position, obj_center)
    pano_all_4K_path = file_path+'/pano/image/' + 'pano_all_4k.png'
    pano_all_depth_4K_path = file_path+'/pano/' + 'pano_depth_all_4k.npy'
    pano_wall_4K_path = file_path+'/pano/image/' + 'pano_wall_4k.png'
    pano_wall_depth_4K_path = file_path+'/pano/' + 'pano_depth_wall_4k.npy'

    c2w = get_four_ador_pose(mesh_obj, room_bound, scale=0.7)

    save_init_depth(c2w_ori, mesh_obj, room_obj, mesh_total_obj,
                    obj_file_path, pano_all_4K_path, pano_all_depth_4K_path, rot=rot)

    angle_distance_list = []
    new_c2w = []

    for i in range(len(c2w)):
        angle_distance = rot_error(
            c2w_ori[0, :3, :3].numpy(), c2w[i, :3, :3].numpy())
        angle_distance_list.append(angle_distance)
    sorted_angle_id = sorted(range(len(angle_distance_list)),
                             key=lambda k: angle_distance_list[k], reverse=False)
    for i in range(len(sorted_angle_id)):
        new_c2w.append(c2w[sorted_angle_id[i]])
    new_c2w = torch.stack(new_c2w)

    for i in tqdm(range(len(new_c2w))):
        save_path = obj_file_path + '/pers_0{}/'.format(i)
        save_pers_img(new_c2w[i].unsqueeze(0), mesh_obj, room_obj,
                      pano_wall_4K_path, pano_wall_depth_4K_path, save_path, rot=rot)


def pre_pers_depth(cfg):
    pano_center = cfg['pano']['pano_cam_center']
    mesh_empty_room, mesh_total_obj, mesh_obj_list, mesh_boundry = load_mesh(
        cfg)
    if cfg['adornment']:
        mesh_adornment, mesh_adornment_list = load_adornment(cfg)
        mesh_total_obj = trimesh.util.concatenate(
            [mesh_total_obj, mesh_adornment])

    mesh_empty_room = trans_room_pano(mesh_empty_room, pano_center)
    mesh_boundry = trans_room_pano(mesh_boundry, pano_center)
    room_bound = get_mesh_bound(mesh_empty_room)
    mesh_empty_room_with_b = trimesh.util.concatenate(
        [mesh_empty_room, mesh_boundry])
    mesh_total_obj = trans_room_pano(mesh_total_obj, pano_center)

    for i in range(len(cfg['obj_id'])):
        file_path = cfg['save_path']
        obj_file_path = cfg['save_path'] + '/' + cfg['obj_id'][i]
        # TODO
        # if cfg['obj_id'][i] in ["tv_cabinet"]:
        pre_obj_depth(trans_room_pano(
            mesh_obj_list[i], pano_center), mesh_empty_room_with_b, mesh_total_obj, room_bound, file_path, obj_file_path)

    if cfg['adornment']:
        for i in range(len(cfg['adornment_id'])):
            file_path = cfg['save_path']
            obj_file_path = cfg['save_path'] + '/' + cfg['adornment_id'][i]
            # TODO
            # if cfg['adornment_id'][i] in ["frame1", "scroll", "scroll2", "scroll3", "scroll4", "clock"]:
            pre_ador_depth(trans_room_pano(
                mesh_adornment_list[i], pano_center), mesh_empty_room_with_b, mesh_total_obj, room_bound, file_path, obj_file_path)


def render_room(cfg, obj_path, file_name):
    rot = -cfg['pano']['rot']
    file_path = cfg['save_path']
    if file_name == "init":

        pano_all_4K_path = '{}/pano/image/pano_all_4k.png'.format(file_path)
        pano_all_depth_4K_path = file_path+'/pano/pano_depth_all_4k.npy'
        c2w = np.load('{}/{}/pose.npy'.format(obj_path, file_name))
        c2w = torch.from_numpy(c2w)
        f = float(np.load('{}/{}/init_f.npy'.format(obj_path, file_name)))

        # save initial pano2pers img
        init_img = get_init_rgb_from_pano(
            pano_all_4K_path, pano_all_depth_4K_path, 1, c2w, f=f, rot=rot)
        cv2.imwrite('{}/{}/img_init.png'.format(obj_path, file_name),
                    cv2.cvtColor(init_img, cv2.COLOR_BGR2RGB))
    else:
        pano_wall_4K_path = file_path+'/pano/image/' + 'pano_wall_4k.png'
        pano_wall_depth_4K_path = file_path+'/pano/' + 'pano_depth_wall_4k.npy'

        c2w = np.load('{}/{}/pose.npy'.format(obj_path, file_name))
        c2w = torch.from_numpy(c2w)
        # save empty room img
        img = get_init_rgb_from_pano(
            pano_wall_4K_path, pano_wall_depth_4K_path, 1, c2w, rot=rot)
        cv2.imwrite("{}/{}/room_empty.png".format(obj_path,
                    file_name), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def pre_pers_empty_room(cfg):
    processes = []
    for num in range(len(cfg['obj_id'])):
        # TODO
        # if cfg['obj_id'][num] in ["sink"]:
        obj_path = cfg['save_path'] + '/' + cfg['obj_id'][num]
        files = glob.glob(obj_path + "/*")
        total_pers_num = len(files)
        for i in range(total_pers_num):
            if i == 0:
                f_name = 'init'
            else:
                f_name = 'pers_0{}'.format(i-1)
            process = multiprocessing.Process(
                target=render_room, args=(cfg, obj_path, f_name))
            process.start()
            processes.append(process)

    if cfg['adornment']:
        for num in range(len(cfg['adornment_id'])):
            obj_path = cfg['save_path'] + '/' + cfg['adornment_id'][num]
            files = glob.glob(obj_path + "/*")
            total_pers_num = len(files)
            for i in range(total_pers_num):
                if i == 0:
                    f_name = 'init'
                else:
                    f_name = 'pers_0{}'.format(i-1)
                process = multiprocessing.Process(
                    target=render_room, args=(cfg, obj_path, f_name))
                process.start()
                processes.append(process)

    for process in processes:
        process.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    # pre_pers_depth(cfg)
    # print("Depth Done")
    pre_pers_empty_room(cfg)
    print("Room Done")
