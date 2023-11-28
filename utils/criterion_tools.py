import cv2
from skimage.metrics import structural_similarity
import torch
from torchmetrics.multimodal.clip_score import CLIPScore
import numpy as np
from functools import partial
import os

metric = CLIPScore(
    model_name_or_path="/cpfs01/user/wangqi/openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    # images_int = (images * 255).astype("uint8")
    clip_score = metric(torch.from_numpy(images).permute(2, 0, 1), prompts)
    return round(float(clip_score), 4)


# ssim和clip都是越大越好 这里为了消除两个因素本身范围的不同 将mean和variance控制到了一个范围内
def ssim_clip_pick(img_lst, ori_img, prompts, ms, vs, mc, vc):
    cur_bestscore = -100
    cur_num = 0
    re_id = 0
    for i in img_lst:
        ssim_score = structural_similarity(
            np.array(i), ori_img, channel_axis=2)
        print(ssim_score)
        ssim_score = (ssim_score - ms) / vs
        clip_score = calculate_clip_score(np.array(i), prompts)
        print(clip_score)
        clip_score = (clip_score - mc) / vc
        tot_score = ssim_score + clip_score
        if (tot_score > cur_bestscore):
            cur_bestscore = tot_score
            re_id = cur_num
        cur_num += 1
    return img_lst[re_id], re_id


def pre_ssim(img_lst, ori_img):
    score_lst = []
    for i in img_lst:
        ssim_score = structural_similarity(
            np.array(i), ori_img, channel_axis=2)
        score_lst.append(ssim_score)
    score_arr = np.array(score_lst)
    return np.mean(score_arr), np.std(score_arr)


def pre_clip(img_lst, prompts):
    score_lst = []
    for i in img_lst:
        clip_score = calculate_clip_score(np.array(i), prompts)
        score_lst.append(clip_score)
    score_arr = np.array(score_lst)
    return np.mean(score_arr), np.std(score_arr)

# img1和img2是生成前后的两张图 mask是两张图的overlap部分 mse越小这个分数越大 相当于局部psnr
# 其中mask是个0-1 mask 1的部分是overlap部分


def overlap_psnr_score(img1, img2, mask):
    diff = img1 - img2
    diff[mask == 0] = 0
    num = np.sum(mask)
    diff = diff * diff
    total_sum = np.sum(diff)
    mse = total_sum / num
    max_pixel_value = 255  # 假设图像像素值在0到255之间
    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr


def pre_psnr(ori_img, img_lst, mask):
    l = len(img_lst)
    score_lst = []
    for i in range(l):
        psnr_score = overlap_psnr_score(np.array(img_lst[i]), ori_img, mask)
        score_lst.append(psnr_score)
    score_arr = np.array(score_lst)
    return np.mean(score_arr), np.std(score_arr)


def psnr_clip_pick(img_lst, mask, ori_img, prompts, mp, vp, mc, vc):
    l = len(img_lst)
    cur_bestscore = -10000
    cur_num = 0
    re_id = 0
    for i in range(l):
        psnr_score = overlap_psnr_score(np.array(img_lst[i]), ori_img, mask)
        psnr_score = (psnr_score - mp) / vp
        clip_score = calculate_clip_score(np.array(img_lst[i]), prompts)
        clip_score = (clip_score - mc) / vc
        tot_score = psnr_score + clip_score
        if (tot_score > cur_bestscore):
            cur_bestscore = tot_score
            re_id = cur_num
        cur_num += 1
    return img_lst[re_id], re_id


def pick_init_inpaint(inpaint_list, img_ori_path, text_prompt):
    img = cv2.imread(img_ori_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mean_ssim, var_ssim = pre_ssim(inpaint_list, img)
    mean_clip, var_clip = pre_clip(inpaint_list, text_prompt)
    re_img, re_id = ssim_clip_pick(
        inpaint_list, img, text_prompt, mean_ssim, var_ssim, mean_clip, var_clip)
    return np.array(re_img), re_id


def pick_inpaint(inpaint_list, img_ori, mask, text_prompt):
    mean_clip, var_clip = pre_clip(inpaint_list, text_prompt)
    mean_psnr, var_psnr = pre_psnr(img_ori, inpaint_list, mask)
    re_img, re_id = psnr_clip_pick(
        inpaint_list, mask, img_ori, text_prompt, mean_psnr, var_psnr, mean_clip, var_clip)
    return np.array(re_img), re_id
