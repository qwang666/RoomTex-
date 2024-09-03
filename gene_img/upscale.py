import sys
import io
import os
import base64
import json
from io import BytesIO
import requests
from PIL import Image, PngImagePlugin
import cv2
from utils.file_tools import load_cfg, check_path
import numpy as np
import argparse

# url = "http://127.0.0.1:7861"


def payload_args(img, resize_mode=0, gfpgan=0, code_vis=0, code_w=0, upscale=2, upscaler1="R-ESRGAN 4x+", up2_vis=0):
    payload = {
        "resize_mode": resize_mode,
        "gfpgan_visibility": gfpgan,
        "codeformer_visibility": code_vis,
        "codeformer_weight": code_w,
        "upscaling_resize": upscale,
        "upscaler_1": upscaler1,
        "upscaler_2": "None",
        "extras_upscaler_2_visibility": up2_vis,
        "image": img
    }
    return payload


def re_img_from_pth(url, img_pth, upscale=2):
    img = cv2.imread(img_pth)
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    payload = payload_args(encoded_image, upscale=upscale)
    # print(payload)
    response = requests.post(
        url=f'{url}/sdapi/v1/extra-single-image', json=payload)

    # Read results
    r = response.json()
    result = r['image']
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    return image


def re_img(url, img, upscale=2):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    payload = payload_args(encoded_image, upscale=upscale)
    # print(payload)
    response = requests.post(
        url=f'{url}/sdapi/v1/extra-single-image', json=payload)

    # Read results
    r = response.json()
    result = r['image']
    image = Image.open(io.BytesIO(base64.b64decode(result.split(",", 1)[0])))
    return np.array(image)


def upscale_img_from_pth(url, img_pth, upscale=2):
    image = re_img_from_pth(url, img_pth, upscale)
    return image


def upscale_img(url, img, upscale=2):
    image = re_img(url, img, upscale)
    return image


# image = re_img(url, "final_pano.png")
# image.save('output.png')
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    parser.add_argument('--port', type=str, default='7861',
                        help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    url = "http://127.0.0.1:" + args.port

    save_path = cfg['save_path'] + '/pano/image'
    check_path(save_path)
    img_pth = cfg['img_path']['pano_all_2K']
    image = upscale_img_from_pth(url, img_pth, upscale=2)
    image.save(save_path+"/pano_all_4k.png")

    img_pth = cfg['img_path']['pano_wall_2K']
    image = upscale_img_from_pth(url, img_pth, upscale=2)
    image.save(save_path+"/pano_wall_4k.png")

    # img_pth = 'demo_results/livingroom/pano/image/wall_ori.png'
    # image = upscale_img_from_pth(url, img_pth, upscale=2)
    # image.save(save_path+"/pano_wall_ori_4k.png")
