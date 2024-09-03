import requests
import io
import base64
from PIL import Image
import cv2

# url = "http://127.0.0.1:7861"


def payload_args(override, init_images, mask, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height,
                 batch_size, batch_count, controlnet, refiner_model, switch_point, resize_mode=0, denoising_strength=0.75, mask_blur=0, inpainting_fill=4,
                 inpaint_full_res=False, inpaint_full_res_padding=32, inpainting_mask_invert=0, seed=-1):
    payload = {
        "override_settings": override,
        "init_images": [init_images],
        "mask": mask,
        # ["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
        "resize_mode": resize_mode,
        "denoising_strength": denoising_strength,
        "mask_blur": mask_blur,  # 蒙版模糊 4
        "inpainting_fill": inpainting_fill,  # 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零 4双mask
        # inpaint area, False: whole picture True：only masked
        "inpaint_full_res": inpaint_full_res,
        # Only masked padding, pixels 32
        "inpaint_full_res_padding": inpaint_full_res_padding,
        "inpainting_mask_invert": inpainting_mask_invert,  # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容
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


def re_img(cfg, ori_pth, mask_pth, occ_mask_pth, dep_pth, obj_prompt, url):
    # Read Image in RGB order
    ori_img = cv2.imread(ori_pth)
    # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

    mask_img = cv2.imread(mask_pth)
    # mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    occ_mask_img = cv2.imread(occ_mask_pth)

    dep_img = cv2.imread(dep_pth)
    # dep_img = cv2.cvtColor(dep_img, cv2.COLOR_BGR2RGB)

    # Encode into PNG and send to ControlNet
    retval1, bytes1 = cv2.imencode('.png', ori_img)
    retval2, bytes2 = cv2.imencode('.png', mask_img)
    retval3, bytes3 = cv2.imencode('.png', dep_img)
    retval4, bytes4 = cv2.imencode('.png', occ_mask_img)
    encoded_image_ori = base64.b64encode(bytes1).decode('utf-8')
    encoded_image_mask = base64.b64encode(bytes2).decode('utf-8')
    encoded_image_dep = base64.b64encode(bytes3).decode('utf-8')
    encoded_image_occ_mask = base64.b64encode(bytes4).decode('utf-8')

    controlnet = controlnet_args(
        encoded_image_dep, "none", "diffusers_xl_depth_full [2f51180b]", 1.5, 1, True)
    prompt = cfg['generate']['pers_inpaint']['prompt'] + obj_prompt
    negative_prompt = cfg['generate']['pers_inpaint']['negative_prompt']
    steps = cfg['generate']['pers_inpaint']['steps']
    sampler_name = cfg['generate']['pers_inpaint']['sampler_name']
    cfg_scale = cfg['generate']['pers_inpaint']['cfg_scale']
    width = cfg['generate']['pers_inpaint']['width']
    height = cfg['generate']['pers_inpaint']['height']
    seed = cfg['generate']['pers_inpaint']['seed']
    sd_model = cfg['generate']['pers_inpaint']['sd_model']
    sd_vae = cfg['generate']['pers_inpaint']['sd_vae']
    refiner_model = cfg['generate']['pers_inpaint']['refiner_model']
    switch_point = cfg['generate']['pers_inpaint']['switch_point']
    override_settings = {
        "sd_model_checkpoint": sd_model,
        "sd_vae": sd_vae
    }
    batch_size = cfg['generate']['pers_inpaint']['batch_size']
    batch_count = cfg['generate']['pers_inpaint']['batch_count']

    payload = payload_args(override_settings, encoded_image_ori, [encoded_image_mask, encoded_image_occ_mask], prompt, negative_prompt, steps,
                           sampler_name, cfg, width, height, batch_size, batch_count, controlnet, refiner_model, switch_point, seed)
    # print(payload)
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)
    # Read results
    r = response.json()
    img_lst = []
    results = r['images']
    for i in results:
        img = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        img_lst.append(img)
    return img_lst[:-1]


def gene_init_inpaint_img(cfg, ori_pth, mask_pth, occ_mask_pth, dep_pth, inpaint_gene_path, obj_prompt, url):
    img_lst = re_img(cfg, ori_pth, mask_pth, occ_mask_pth,
                     dep_pth, obj_prompt, url)
    for i in range(len(img_lst)):
        img_lst[i].save(inpaint_gene_path + f"/{i}.png")
    return img_lst


# if __name__ == "__main__":
