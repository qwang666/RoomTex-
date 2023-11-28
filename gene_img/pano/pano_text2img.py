import requests
import io
import base64
from PIL import Image
import cv2
from utils.file_tools import load_cfg, check_path
import argparse

# url = "http://127.0.0.1:7861"


def payload_args(override, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, batch_size, batch_count, controlnet, refiner_model, switch_point, seed=-1):
    payload = {
        "override_settings": override,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "sampler_name": sampler_name,
        "CFG scale": cfg_scale,
        "seed": seed,
        "width": width,
        "height": height,
        "batch_size": batch_size,
        "n_iter": batch_count,
        "refiner_checkpoint": refiner_model,
        "refiner_switch_at": switch_point,
        "alwayson_scripts": {
            "controlnet": controlnet
        }
    }
    return payload
# module对应于预处理模块 none还是midas还是别的
# model对应于所使用的controlnet depth model
# weight对应于强度
# resize_mode对应于just resize, crop and resize, resize and fill
# control mode对应于 balanced, prompt more important, controlnet more important


def controlnet_args(input_img, module, model, weight, resize_mode, low_vram=True, pro_res=512, thre_a=0, thre_b=1, gui_st=0, gui_ed=1, control_mode=0, pixel=True):
    controlnet = {
        "args": [
            {
                "input_image": input_img,
                "module": module,
                "model": model,
                "weight": weight,
                "resize_mode": resize_mode,
                "low_vram": low_vram,
                "processor_res": pro_res,
                "threshold_a": thre_a,
                "threshold_b": thre_b,
                "guidance_start": gui_st,
                "guidance_end": gui_ed,
                "control_mode": control_mode,
                "pixel_perfect": pixel
            }
        ]
    }
    return controlnet


def re_img(cfg, depth_pth, url):
    img = cv2.imread(depth_pth)
    retval, bytes = cv2.imencode('.png', img)
    encoded_image = base64.b64encode(bytes).decode('utf-8')

    controlnet = controlnet_args(
        encoded_image, "none", "diffusers_xl_depth_full [2f51180b]", 1, 1, True)
    prompt = cfg['generate']['pano']['prompt']
    negative_prompt = cfg['generate']['pano']['negative_prompt']
    steps = cfg['generate']['pano']['steps']
    sampler_name = cfg['generate']['pano']['sampler_name']
    cfg_scale = cfg['generate']['pano']['cfg_scale']
    width = cfg['generate']['pano']['width']
    height = cfg['generate']['pano']['height']
    seed = cfg['generate']['pano']['seed']
    sd_model = cfg['generate']['pano']['sd_model']
    sd_vae = cfg['generate']['pano']['sd_vae']
    refiner_model = cfg['generate']['pano']['refiner_model']
    switch_point = cfg['generate']['pano']['switch_point']
    override_settings = {
        "sd_model_checkpoint": sd_model,
        "sd_vae": sd_vae
    }
    batch_size = cfg['generate']['pano']['batch_size']
    batch_count = cfg['generate']['pano']['batch_count']

    payload = payload_args(override_settings, prompt, negative_prompt, steps, sampler_name, cfg,
                           width, height, batch_size, batch_count, controlnet, refiner_model, switch_point, seed)
    # print(payload)
    response = requests.post(url=f'{url}/sdapi/v1/txt2img', json=payload)

    # Read results
    r = response.json()
    img_lst = []
    results = r['images']
    for i in results:
        img = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        img_lst.append(img)
    return img_lst


def gene_text2img(cfg, url):
    depth_path = cfg['save_path'] + '/pano' + '/pano_all_disp_2k.png'
    image = re_img(cfg, depth_path, url)
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    parser.add_argument('--port', type=str, default='7861',
                        help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    url = "http://127.0.0.1:" + args.port

    image = gene_text2img(cfg, url)
    save_path = cfg['save_path'] + '/pano/image'
    check_path(save_path)
    for i in range(len(image)):
        image[i].save(save_path + f"/{i}.png")
