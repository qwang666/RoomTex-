import glob
import numpy as np
import torch
import cv2
from utils.file_tools import check_path, load_cfg
from utils.world_points import read_imgx4_depth_mask, img2world, world2img, get_cv_sd_mask, get_combine_img
from gene_img.pers.pers_inpaint import gene_inpaint_pers_img
from gene_img.pers.init_inpaint import gene_init_inpaint_img
from utils.criterion_tools import pick_inpaint, pick_init_inpaint
from gene_img.upscale import upscale_img
import argparse
import time


def project_new_view(cfg, obj_path, obj_prompt, url, obj_id):
    files = glob.glob(obj_path + "/*")
    total_pers_num = len(files)
    scale = 4
    img_list = []
    world_coords_list = []
    mask_list = []
    choose_img_path = '{}/init/choose_imgs/'.format(obj_path)
    check_path(choose_img_path)
    for i in range(total_pers_num-1):
        if i == 0:
            f_name = 'init'
            f = np.load(obj_path + '/' + f_name + '/init_f.npy')
            img_init_pth = obj_path + '/' + f_name + '/img_init.png'
            sd_mask_pth = obj_path + '/' + f_name + '/sd_mask.png'
            disp_pth = obj_path + '/' + f_name + '/disp_all.png'
            occ_mask_pth = obj_path + '/' + f_name + '/occ_mask.png'
            inpaint_gene_path = obj_path + '/' + f_name + "/inpaint_imgs/"
            check_path(inpaint_gene_path)
            ######
            # img_init_pth = ''
            # ori_img = cv2.imread(img_init_pth)
            # inpaint_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
            #######
            inpaint_list = gene_init_inpaint_img(
                cfg, img_init_pth, sd_mask_pth, occ_mask_pth, disp_pth, inpaint_gene_path, obj_prompt, url)
            inpaint_img, re_id = pick_init_inpaint(
                inpaint_list, img_init_pth, cfg['generate']['pers_inpaint']['prompt'] + obj_prompt)

            cv2.imwrite('{}/img_0_init.png'.format(choose_img_path),
                        cv2.cvtColor(inpaint_img, cv2.COLOR_RGB2BGR))
            inpaint_img = upscale_img(url, inpaint_img, 4)
            img_path = obj_path + '/' + f_name + '/inpaint.png'
            cv2.imwrite(img_path, cv2.cvtColor(inpaint_img, cv2.COLOR_RGB2BGR))
            print("choose img_{} as init inpaint".format(re_id))

            depth_path = obj_path + '/' + f_name + '/depth_all_4k.npy'
            mask_path = obj_path + '/' + f_name + '/mask_obj_4k.png'
            c2w = np.load(obj_path + '/' + f_name + '/pose.npy')
            img, depth, mask, intr = read_imgx4_depth_mask(
                img_path, depth_path, mask_path, scale, kernel_size=9, iter_num=2, f=f)
        else:
            f_name = 'pers_0{}'.format(i-1)
            img_path = obj_path + '/' + f_name + '/inpaint.png'
            depth_path = obj_path + '/' + f_name + '/depth_all_4k.npy'
            mask_path = obj_path + '/' + f_name + '/mask_obj_4k.png'
            proj_mask_path = obj_path + '/' + f_name + '/proj_mask.png'
            c2w = np.load(obj_path + '/' + f_name + '/pose.npy')
            img, depth, mask, intr = read_imgx4_depth_mask(
                img_path, depth_path, mask_path, scale, kernel_size=9, iter_num=2, init=False, sd_mask_path=proj_mask_path)
        print(f_name)
        f_next_name = 'pers_0{}'.format(i)
        world_coords = img2world(img, depth, mask, intr, torch.from_numpy(c2w))
        img_list.append(img.reshape(-1, 3))
        world_coords_list.append(world_coords)
        mask_list.append(mask)

        total_img = np.concatenate(img_list, axis=0)
        total_world_coords = torch.cat(world_coords_list, dim=-1)
        total_mask = np.concatenate(mask_list, axis=-1)

        # new pose view
        new_poses = np.load(obj_path + '/' + f_next_name + '/pose.npy')
        img_warp, img_warp_depth = world2img(torch.from_numpy(
            new_poses), total_img, total_world_coords, total_mask, new_w=1024, new_h=1024, f=500.0)
        depth_gt_path = obj_path + '/' + f_next_name + '/depth_all.npy'
        depth_gt = np.load(depth_gt_path)
        depth_mask = (img_warp_depth-0.003) < depth_gt
        img_depth_warp = img_warp*depth_mask[:, :, np.newaxis]

        # Combine imgs
        inpaint_pano_img_path = obj_path + '/' + f_next_name + '/room_empty.png'
        inpaint_pano_img = cv2.imread(inpaint_pano_img_path)
        inpaint_pano_img = cv2.cvtColor(inpaint_pano_img, cv2.COLOR_BGR2RGB)

        new_mask_obj = cv2.imread(
            obj_path + '/' + f_next_name + '/mask_obj.png')
        cv_mask, sd_mask, sd_fill = get_cv_sd_mask(
            img_depth_warp, depth_gt, new_mask_obj, kernel_size=5, itera=2)
        img_depth_warp_cv2 = cv2.inpaint(img_depth_warp.astype(
            np.uint8), cv_mask[:, :, 0], 3, cv2.INPAINT_TELEA)
        combine_render, combine_pers, combine_img = get_combine_img(
            inpaint_pano_img, img_depth_warp_cv2, sd_mask, new_mask_obj, sd_fill)
        dilate_sd_mask = cv2.dilate(
            sd_mask, np.ones((5, 5), np.uint8), iterations=1)

        sd_img_pth = obj_path + '/' + f_next_name + "/sd_img.png"
        sd_mask_pth = obj_path + '/' + f_next_name + "/sd_mask.png"
        cv2.imwrite(sd_mask_pth, dilate_sd_mask)
        cv2.imwrite(obj_path + '/' + f_next_name+"/proj_mask.png", cv_mask)
        cv2.imwrite(sd_img_pth, cv2.cvtColor(combine_img, cv2.COLOR_BGR2RGB))

        disp_pth = obj_path + '/' + f_next_name+"/disp_all.png"
        inpaint_gene_path = obj_path + '/' + f_next_name + "/inpaint_imgs/"
        check_path(inpaint_gene_path)
        inpaint_list = gene_inpaint_pers_img(
            cfg, sd_img_pth, sd_mask_pth, disp_pth, inpaint_gene_path, obj_prompt, url)

        overlap_mask = dilate_sd_mask - sd_mask
        cv2.imwrite(obj_path + '/' + f_next_name +
                    "/overlap_mask.png", overlap_mask)
        inpaint_img, re_id = pick_inpaint(
            inpaint_list, combine_img, overlap_mask, cfg['generate']['pers_inpaint']['prompt'] + obj_prompt)
        cv2.imwrite('{}/img_{}.png'.format(choose_img_path, i),
                    cv2.cvtColor(inpaint_img, cv2.COLOR_RGB2BGR))
        inpaint_img = upscale_img(url, inpaint_img, 4)
        cv2.imwrite(obj_path + '/' + f_next_name + "/inpaint.png",
                    cv2.cvtColor(inpaint_img, cv2.COLOR_RGB2BGR))
        print("{} inpaint finished. choose img_{}".format(f_name, re_id))
    print("Obj generate finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    parser.add_argument('--port', type=str, default='7861', help='url port')
    parser.add_argument('--id', type=int, default='0', help='obj id')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    url = "http://127.0.0.1:" + args.port
    num = args.id
    # num = cfg['generate']['id']
    obj_path = cfg['save_path'] + '/' + cfg['obj_id'][num]
    obj_prompt = cfg['generate']['pers_inpaint']['prompt'] + \
        cfg['obj_describe'][num]

    start = time.perf_counter()
    project_new_view(cfg, obj_path, obj_prompt, url, cfg['obj_id'][num])
    end = time.perf_counter()
    print(cfg['obj_id'][num])
    print("Time: ", round((end-start)/60), 'min ',
          round((end-start) % 60), 'seconds')
