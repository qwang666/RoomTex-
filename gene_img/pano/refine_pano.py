#这个文件是获得了panorama后处理所有周围环境并且填补地板的代码
from diffusers import StableDiffusionInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
from utils.pipeline_stable_diffusion_controlnet_inpaint import *
from diffusers.utils import load_image

import cv2
from PIL import Image
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
from utils.file_tools import load_cfg, check_path
import trimesh
from utils.cameras.cameras import Cameras, CameraType
from trimesh.ray.ray_pyembree import RayMeshIntersector
import json
import requests
import io
import base64

def get_pix_coords(height, width):
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))
    return xyz # [3, H*W]

def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))
def get_poses(position, centers):
    forward_vector = safe_normalize(centers-position)
    up_vector = torch.FloatTensor([0, 0.0, 1.0]).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(position.shape[0], 1, 1)
    poses[:, :3, :3] = torch.stack((right_vector, -up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = position
    return poses

def img2world(img, depth, mask, intr, c2w):
    height, width = depth.shape
    image_coords = get_pix_coords(height, width)
    inv_intr = torch.inverse(intr)
    camera_coords = torch.from_numpy(depth).view(1, -1) * image_coords
    camera_coords = torch.matmul(inv_intr, camera_coords.float())  # [3, H*W].
    camera_coords = torch.cat((camera_coords, torch.ones((1, height*width))), dim = 0).float()

    world_coords = torch.matmul(c2w, camera_coords)
    return world_coords

def interpolate_depth(depth, scale):
    h, w = depth.shape
    depth = torch.nn.functional.interpolate(torch.tensor(depth[None,None,:,:]), scale_factor=scale, mode='bilinear', align_corners=True)
    depth = depth.squeeze(0).squeeze(0).numpy()

    return depth

def read_scale_img_depth_mask(img, depth, mask, scale, kernel=np.ones((7, 7), np.uint8), iter=2, fx=500.0, fy=500.0):
    depth = np.array(depth)
    depth = interpolate_depth(depth, scale)
    height, width = depth.shape

    mask = np.array(mask)
    
    for i in range(iter):
        mask = cv2.erode(mask, kernel, 2)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:,:,-1]

    mask = mask.reshape(-1) > 125
    img = np.array(img)
    img = cv2.resize(img, (width, height))

    cx=(width-1)/2
    cy=(height-1)/2
    fx=fx*scale
    fy=fy*scale

    intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])

    return img, depth, mask, intr

def inpaint_floor(pano_path, mask_obj_path, depth_path, kernel_size, iter, text_prompt):
    init_image = Image.open(pano_path).convert("RGB")
    init_img_arr = np.array(init_image)
    init_mask = Image.open(mask_obj_path).convert("RGB")
    empty_depth = Image.open(depth_path).convert("RGB")
    
    mask_obj_arr = np.array(init_mask)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    for i in range(iter):
        mask_obj_arr = cv2.dilate(mask_obj_arr, kernel, 5)
    #这张是dilate之后的mask
    dilated_mask = Image.fromarray(mask_obj_arr)
    #这张是被遮了之后的img
    masked_img_arr = init_img_arr
    masked_img_arr[np.where(mask_obj_arr == 255)] = 255
    masked_img = Image.fromarray(masked_img_arr)
    
    #下面构建inpainting的pipeline
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')


    negative_prompt = 'anime, cartoon, graphic, text, painting, crayon, graphite, abstract glitch, blurry, low quality, worst quality, inconsistent, rough surface'
    new_image = pipe(
        text_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        image=init_image,
        control_image=empty_depth,
        mask_image=dilated_mask,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.3
    ).images[0]

    return masked_img, new_image

def inpaint_loop(pano_img, depth_path, output_path, text_prompt):
    ori_pano_arr = np.array(pano_img)
    H, W = ori_pano_arr.shape[0], ori_pano_arr.shape[1]
    #下面是把图片切开来重新拼起来
    pz_image = np.zeros_like(ori_pano_arr)
    pz_image[:, 0:W//2] = ori_pano_arr[:, W//2:]
    pz_image[:, W//2:] = ori_pano_arr[:, 0:W//2]
    stitched_img = Image.fromarray(pz_image)
    #下面在边界处取一个固定比例的mask 为宽度的1/10 然后把中间部分mask掉用于重新inpainting
    tmp_width = int(W/30)
    
    mask = np.zeros_like(ori_pano_arr)
    mask[:, W//2 - tmp_width:W//2 + tmp_width] = [255, 255, 255]
    pz_image[:, W//2 - tmp_width:W//2 + tmp_width] = [255, 255, 255]
    masked_i = Image.fromarray(pz_image)
    mask_image = Image.fromarray(mask)
    
    #下面读入depth图用于inpainting,并且要对于depth做和pano一样的操作
    depth_img = Image.open(depth_path).convert('RGB')
    dp_wall = np.array(depth_img)
    new_depth = np.zeros_like(dp_wall)
    H, W = new_depth.shape[0], new_depth.shape[1]
    new_depth[:, 0:W//2] = dp_wall[:, W//2:]
    new_depth[:, W//2:] = dp_wall[:, 0:W//2]
    new_depth_image = Image.fromarray(new_depth.astype(np.uint8))

    # speed up diffusion process with faster scheduler and memory optimization
    controlnet = ControlNetModel.from_pretrained(
        "fusing/stable-diffusion-v1-5-controlnet-depth", torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        "runwayml/stable-diffusion-inpainting", controlnet=controlnet, torch_dtype=torch.float16
    )

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    # Remove if you do not have xformers installed
    # see https://huggingface.co/docs/diffusers/v0.13.0/en/optimization/xformers#installing-xformers
    # for installation instructions
    pipe.enable_xformers_memory_efficient_attention()

    pipe.to('cuda')


    negative_prompt = 'anime, graphic, text, painting, crayon, graphite, abstract glitch, blurry, low quality, worst quality, inconsistent, rough surface'
    new_image = pipe(
        text_prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=50,
        image=stitched_img,
        control_image=new_depth_image,
        mask_image=mask_image,
        guidance_scale=7.5
    ).images[0]

    new_img_arr = np.array(new_image)
    final_arr = np.zeros_like(new_img_arr)
    final_arr[:, 0:W//2] = new_img_arr[:, W//2:]
    final_arr[:, W//2:] = new_img_arr[:, 0:W//2]
    final_img = Image.fromarray(final_arr)
    final_img.save(output_path)
    
    return masked_i, final_img



#输入是房间的内壁mesh 输出向上看向下看的两个depth用于后续进行inpainting
def get_poses1(position, centers):
    forward_vector = safe_normalize(position-centers)
    up_vector = torch.FloatTensor([0, 1.0, 0.0]).unsqueeze(0)
    right_vector = safe_normalize(torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(0).repeat(position.shape[0], 1, 1)
    poses[:, :3, :3] = torch.stack((-right_vector, up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = position
    return poses

def wall_depth(pose, mesh):
    c2w = pose[0,:3,:]
    width_pers = 1024
    height_pers = 1024
    cx=(width_pers-1)/2
    cy=(height_pers-1)/2
    fx=500.0
    fy=500.0

    perspective_camera = Cameras(
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        width = width_pers,
        height = height_pers,
        camera_to_worlds=c2w,
        camera_type=CameraType.PERSPECTIVE,
    )
    camera_ray_bundle = perspective_camera.generate_rays(camera_indices=0)
    num_steps = 256
    bound = 3
    min_near = 0.1
    perturb = True
    rays_o = camera_ray_bundle.origins # [h, w, 3]
    rays_d = camera_ray_bundle.directions # [h, w, 3]
    rays_o = rays_o.view(-1, 3) # [h*w, 3]
    rays_d = rays_d.view(-1, 3) # [h*w, 3]

    points, index_ray, _ = RayMeshIntersector(mesh).intersects_location(rays_o, rays_d, multiple_hits=False)

    coords = np.array(list(np.ndindex(height_pers, width_pers))).reshape(height_pers, width_pers,-1).reshape(-1,2) #.transpose(1,0,2).reshape(-1,2)
    c2w = torch.concat((c2w, torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)), dim=0)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    depth = np.abs((torch.inverse(c2w).numpy() @ points.T)[2])
    pixel_ray = coords[index_ray]
    depthmap_pers = np.full([height_pers, width_pers], 0.1)
    depthmap_pers[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    return depthmap_pers



def prepare_refine_depth(wall_pth, center, pos1, pos2):
    mesh_wall = trimesh.load(wall_pth, force='mesh')
    mesh_v = mesh_wall.vertices
    shift = torch.tensor([[(mesh_v[:,0].min() + mesh_v[:,0].max())/2, (mesh_v[:,1].min() + mesh_v[:,1].max())/2, (mesh_v[:,2].min() + mesh_v[:,2].max())/2]])
    centers = torch.tensor([[center[0], center[2], -center[1]]])
    p1 = torch.tensor([[pos1[0], pos1[2], -pos1[1]]])
    p2 = torch.tensor([[pos2[0], pos2[2], -pos2[1]]])
    centers += shift
    p1 += shift
    p2 += shift
    pose1 = get_poses1(p1, centers)
    pose2 = get_poses1(p2, centers)
    top_down_depth = wall_depth(pose1, mesh_wall)
    bottom_up_depth = wall_depth(pose2, mesh_wall)
    floor_depth = top_down_depth
    floor_depth[np.where(top_down_depth <= 0.15)] = 10
    floor_depth = 1/floor_depth

    floor_depth = np.repeat(floor_depth[:, :, None], 3, axis=2)
    floor_depth = (floor_depth / floor_depth.max() * 255).astype(np.uint8)
    floor_depth = Image.fromarray(floor_depth)
    
    ceiling_depth = bottom_up_depth
    ceiling_depth[np.where(bottom_up_depth <= 0.15)] = 10
    ceiling_depth = 1/ceiling_depth

    ceiling_depth = np.repeat(ceiling_depth[:, :, None], 3, axis=2)
    ceiling_depth = (ceiling_depth / ceiling_depth.max() * 255).astype(np.uint8)
    ceiling_depth = Image.fromarray(ceiling_depth)
    
    return top_down_depth, bottom_up_depth, floor_depth, ceiling_depth
    
    



def prepare_floor(pano_depth, pano_img, rot, position, centers, fx=500.0, fy=500.0):
    pano_disp = np.load(pano_depth)
    pano_img = np.array(pano_img)
    height, width = pano_img.shape[:2]

    pano_img = cv2.resize(pano_img, (int(width), int(height)), interpolation=cv2.INTER_LINEAR)
    pixel_offset = 0.5
    # projection

    image_coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    image_coords = (torch.stack(image_coords, dim=-1) + pixel_offset).float()  # stored as (y, x) coordinates

    s_x = image_coords[:,:,1]*2*torch.pi/width - torch.pi
    s_y = -image_coords[:,:,0]*torch.pi/height + torch.pi/2

    pano_depth = torch.from_numpy(pano_disp).float()
    w_x = pano_depth * torch.sin(s_x) * torch.cos(s_y)
    w_y = pano_depth * torch.cos(s_x) * torch.cos(s_y)
    w_z = pano_depth * torch.sin(s_y)

    wall_world_coords = torch.stack((w_x.view(-1), w_y.view(-1), w_z.view(-1), torch.ones_like(w_x.view(-1))), dim=0).unsqueeze(0)
    #wall_world_coords为panorama的坐标 总共2048*4096个

    wall_mask = np.ones(wall_world_coords.shape[2])

    #下面为投影到某个角度下的照片的代码
    c2w = torch.tensor([[np.cos(rot), np.sin(rot), 0],
                        [-np.sin(rot), np.cos(rot), 0],
                        [0, 0, 1]]).float()
    position = position @ c2w
    centers = centers @ c2w
    pose = get_poses(position, centers)

    img_w = 1024
    img_h = 1024
    cx=(img_w-1)/2
    cy=(img_h-1)/2

    intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])

    c2w_warp = pose[0]
    camera_coords = torch.inverse(c2w_warp) @ wall_world_coords

    image_coords = intr @ camera_coords[:,0:3,:]
    mask = image_coords[:, 2] > 1e-3
    z = image_coords[:, 2]
    z[z<1e-3] = 1e-3

    u = image_coords[:,0]/z
    v = image_coords[:, 1]/z

    mask = (u>=0) & (u<(img_h-0.5)) & (v>=0) & (v<(img_w-0.5)) & mask
    img_warp = np.zeros((img_h, img_w, 3))

    u_view = torch.round(u[mask].view(-1)).int().numpy()
    v_view = torch.round(v[mask].view(-1)).int().numpy()

    total_masked_img = (pano_img.reshape(-1, 3))[mask.view(-1)]
    z_depth = (z.view(-1))[mask.view(-1)]
    img_warp_depth = np.ones((img_h, img_w))*1000
    for i in range(v_view.shape[0]):
            if img_warp_depth[v_view[i], u_view[i]] > z_depth[i]:
                    img_warp[v_view[i], u_view[i]] = total_masked_img[i]
                    img_warp_depth[v_view[i], u_view[i]] = z_depth[i]


    new_img = img_warp.astype(np.uint8)
    mask = new_img.sum(-1) == 0
    mask = mask.astype(np.uint8)
    rendered_image = cv2.inpaint(new_img, mask, 3, cv2.INPAINT_TELEA)
    rendered_image1 = Image.fromarray(rendered_image.astype(np.uint8))

    # 以上是投影到地板的图片的代码
    mask_floor = np.zeros_like(rendered_image)
    H, W = mask_floor.shape[0], mask_floor.shape[1]
    mask_floor[int(H*0.12):int(H*0.88), int(H*0.03):int(H*0.97)] = [255, 255, 255]
    mask_floor_img = Image.fromarray(mask_floor)

    rendered_image[int(H*0.12):int(H*0.88), int(H*0.03):int(H*0.97)] = [255, 255, 255]
    masked_floor_img = Image.fromarray(rendered_image.astype(np.uint8))
    
    return rendered_image1, mask_floor_img, masked_floor_img

def payload_args(override, init_images, mask, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, batch_size, batch_count, controlnet, refiner_model, switch_point, resize_mode=0, denoising_strength=0.75, mask_blur=0, inpainting_fill=1,
                 inpaint_full_res=False, inpaint_full_res_padding=32, inpainting_mask_invert=0, seed=-1):
    payload = {
        "override_settings": override,
        "init_images": [init_images],
        "mask": mask,
        "resize_mode": resize_mode, #["Just resize", "Crop and resize", "Resize and fill", "Just resize (latent upscale)"]
        "denoising_strength": denoising_strength,
        "mask_blur": mask_blur, #蒙版模糊 4
        "inpainting_fill": inpainting_fill,# 蒙版遮住的内容， 0填充， 1原图 2潜空间噪声 3潜空间数值零
        "inpaint_full_res": inpaint_full_res, # inpaint area, False: whole picture True：only masked
        "inpaint_full_res_padding": inpaint_full_res_padding, # Only masked padding, pixels 32
        "inpainting_mask_invert": inpainting_mask_invert, # 蒙版模式 0重绘蒙版内容 1 重绘非蒙版内容
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
#module对应于预处理模块 none还是midas还是别的
#model对应于所使用的controlnet depth model
#weight对应于强度
#resize_mode对应于just resize, crop and resize, resize and fill
#control mode对应于 balanced, prompt more important, controlnet more important
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

def re_img(cfg, url, ori_img, mask_img, dep_img, floor):
    # Read Image in RGB order
    ori_img = np.array(ori_img)
    mask_img = np.array(mask_img)
    dep_img = np.array(dep_img)

    # Encode into PNG and send to ControlNet
    retval1, bytes1 = cv2.imencode('.png', ori_img)
    retval2, bytes2 = cv2.imencode('.png', mask_img)
    retval3, bytes3 = cv2.imencode('.png', dep_img)
    encoded_image_ori = base64.b64encode(bytes1).decode('utf-8')
    encoded_image_mask = base64.b64encode(bytes2).decode('utf-8')
    encoded_image_dep = base64.b64encode(bytes3).decode('utf-8')
    
    controlnet = controlnet_args(encoded_image_dep, "none", "diffusers_xl_depth_full [2f51180b]", 0.9, 1, True)
    prompt = cfg['generate'][floor]['prompt']
    negative_prompt = cfg['generate'][floor]['negative_prompt']
    steps = cfg['generate'][floor]['steps']
    sampler_name = cfg['generate'][floor]['sampler_name']
    cfg_scale = cfg['generate'][floor]['cfg_scale']
    seed = cfg['generate'][floor]['seed']
    width = cfg['generate'][floor]['width']
    height = cfg['generate'][floor]['height']
    sd_model = cfg['generate'][floor]['sd_model']
    sd_vae = cfg['generate'][floor]['sd_vae']
    refiner_model = cfg['generate'][floor]['refiner_model']
    switch_point = cfg['generate'][floor]['switch_point']
    override_settings={
        "sd_model_checkpoint": sd_model,
        "sd_vae": sd_vae
    }
    batch_size = cfg['generate'][floor]['batch_size']
    batch_count = cfg['generate'][floor]['batch_count']
    payload = payload_args(override_settings, encoded_image_ori, encoded_image_mask, prompt, negative_prompt, steps, sampler_name, cfg_scale, width, height, batch_size, batch_count, controlnet, refiner_model, switch_point, seed)
    response = requests.post(url=f'{url}/sdapi/v1/img2img', json=payload)

    # Read results
    r = response.json()
    img_lst = []
    results = r['images']
    for i in results:
        img = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
        img_arr = np.array(img)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img_arr)
        img_lst.append(img)
    return img_lst[0]

##下面这一段是top-down补地板的代码
def floor_to_pano(rot, position, centers, img, depth, mask, fx, fy):
    c2w = torch.tensor([[np.cos(rot), np.sin(rot), 0],
                        [-np.sin(rot), np.cos(rot), 0],
                        [0, 0, 1]]).float()
    position = position @ c2w
    centers = centers @ c2w
    pose = get_poses(position, centers)
    top_down_arr = np.array(img)
    top_down_mask_arr = np.array(mask)
    top_down_arr[np.where(top_down_mask_arr == 0)] = 0
    new_top_down = Image.fromarray(top_down_arr)
    c2w = pose[0].unsqueeze(0)
    s = 6
    imgb_1, depthb_1, maskb_1, intr = read_scale_img_depth_mask(new_top_down, depth, mask, scale=s, kernel=np.ones((7, 7), np.uint8), iter=0, fx=fx, fy=fy)
    world_coordsb_1 = img2world(imgb_1, depthb_1, maskb_1, intr, c2w)

    height = 1024
    width = 2048
    pano_new = np.zeros((1024, 2048, 3))
    xx = world_coordsb_1[0,0]
    yy = world_coordsb_1[0,1]
    zz = world_coordsb_1[0,2]
    pano_y = ((-np.arctan2(zz, np.sqrt(xx ** 2 + yy ** 2)) + (np.pi/2))/(np.pi) * height).int()
    pano_x = ((np.arctan2(xx, yy) + np.pi) / (2*np.pi) * width).int()
    image_cor = torch.meshgrid(torch.arange(height*s), torch.arange(height*s), indexing="ij")
    ori_1 = image_cor[0].reshape(-1)
    ori_2 = image_cor[1].reshape(-1)
    pano_new[pano_y, pano_x] = imgb_1[ori_1, ori_2]
    pano_new_image = Image.fromarray(pano_new.astype(np.uint8))
    pano_new = np.array(pano_new_image)
    mask = pano_new.sum(-1) == 0
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    final_img = cv2.inpaint(pano_new, mask, 3, cv2.INPAINT_TELEA)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    final_img[np.where(mask == 1)] = [0, 0, 0]

    final_save_img = Image.fromarray(final_img)
    return final_img, final_save_img

##下面这一段是bottom-up补天花板的代码
def ceiling_to_pano(rot, position, centers, img, depth, mask, fx, fy):
    c2w = torch.tensor([[np.cos(rot), np.sin(rot), 0],
                        [-np.sin(rot), np.cos(rot), 0],
                        [0, 0, 1]]).float()
    position = position @ c2w
    centers = centers @ c2w
    pose = get_poses(position, centers)
    top_down_arr = np.array(img)
    top_down_mask_arr = np.array(mask)
    top_down_arr[np.where(top_down_mask_arr == 0)] = 0
    new_top_down = Image.fromarray(top_down_arr)
    c2w = pose[0].unsqueeze(0)
    s = 4
    imgb_2, depthb_2, maskb_2, intr = read_scale_img_depth_mask(new_top_down, depth, mask, scale=s, kernel=np.ones((7, 7), np.uint8), iter=0, fx=fx, fy=fy)
    world_coordsb_2 = img2world(imgb_2, depthb_2, maskb_2, intr, c2w)
    print(imgb_2.shape)

    height = 1024
    width = 2048
    pano_new = np.zeros((1024, 2048, 3))
    xx = world_coordsb_2[0,0]
    yy = world_coordsb_2[0,1]
    zz = world_coordsb_2[0,2]
    pano_y = ((-np.arctan2(zz, np.sqrt(xx ** 2 + yy ** 2)) + (np.pi/2))/(np.pi) * height).int()
    pano_x = ((np.arctan2(xx, yy) + np.pi) / (2*np.pi) * width).int()
    image_cor = torch.meshgrid(torch.arange(height*s), torch.arange(height*s), indexing="ij")
    ori_1 = image_cor[0].reshape(-1)
    ori_2 = image_cor[1].reshape(-1)
    pano_new[pano_y, pano_x] = imgb_2[ori_1, ori_2]
    pano_new_image = Image.fromarray(pano_new.astype(np.uint8))
    pano_new = np.array(pano_new_image)
    mask = pano_new.sum(-1) == 0
    mask = mask.astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    final_img = cv2.inpaint(pano_new, mask, 5, cv2.INPAINT_TELEA)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.erode(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    mask = cv2.dilate(mask, kernel, 5)
    final_img[np.where(mask == 1)] = [0, 0, 0]

    mask = np.zeros((1024, 2048)).astype(np.uint8)
    mask[:10,:] = 255
    final_img = cv2.inpaint(final_img, mask, 5, cv2.INPAINT_TELEA)
    final_save_img = Image.fromarray(final_img)
    return final_img, final_save_img

def integrate(ori_pano, top_down_pano, bottom_up_pano):
    ori_pano_arr = np.array(ori_pano)
    top_down_pano_arr = np.array(top_down_pano)
    bottom_up_pano_arr = np.array(bottom_up_pano)
    final_pano_arr = np.zeros_like(top_down_pano)
    final_pano_arr = top_down_pano_arr
    final_pano_arr[np.where(bottom_up_pano_arr != 0)] = bottom_up_pano_arr[np.where(bottom_up_pano_arr != 0)]
    index = np.all(top_down_pano_arr == [0, 0, 0], axis=-1)
    final_pano_arr[index] = ori_pano_arr[index]
    final_pano_img = Image.fromarray(final_pano_arr)
    return final_pano_img

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    parser.add_argument('--port', type=str, default='7861',
                        help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    url = "http://127.0.0.1:" + args.port
    img_pth = cfg['img_path']['pano_all_2K']
    mask_pth = cfg['save_path'] + '/pano/pano_wall_obj_2k.png'
    depth_pth = cfg['save_path'] + '/pano/pano_wall_disp_2k.png'
    pano_depth_pth = cfg['save_path'] + '/pano/pano_depth_wall_2k.npy'
    prompt = cfg['generate']['refine']['prompt']
    out_pth = cfg['img_path']['pano_tmp_wall_2K']
    wall_pth = cfg['room_mesh_path']
    center = torch.tensor(cfg['generate']['refine']['center'])
    pos1 = torch.tensor(cfg['generate']['refine']['pos1'])
    pos2 = torch.tensor(cfg['generate']['refine']['pos2'])
    rot = cfg['pano']['rot']
    masked_img_floor, inpainted_img_floor = inpaint_floor(img_pth, mask_pth, depth_pth, 7, 6, prompt)
    masked_stitched_floor, inpainted_stitched_floor = inpaint_loop(inpainted_img_floor, depth_pth, out_pth, prompt)
    floor_depth, ceiling_depth, floor_disp_image, ceiling_disp_image = prepare_refine_depth(wall_pth, center, pos1, pos2)
    center = center.unsqueeze(0)
    pos1 = pos1.unsqueeze(0)
    pos2 = pos2.unsqueeze(0)
    fx = cfg['generate']['floor']['fx']
    fy = cfg['generate']['floor']['fy']
    pro_floor, mask_floor, masked_img_floor = prepare_floor(pano_depth_pth, inpainted_stitched_floor, rot, pos1, center, fx, fy)
    pro_ceiling, mask_ceiling, masked_img_ceiling = prepare_floor(pano_depth_pth, inpainted_stitched_floor, rot, pos2, center, fx, fy)
    img2img_floor = re_img(cfg, url, pro_floor, mask_floor, floor_disp_image, 'floor')
    img2img_ceiling = re_img(cfg, url, pro_ceiling, mask_ceiling, ceiling_disp_image, 'ceiling')
    floor_pano, floor_pano_img = floor_to_pano(rot, pos1, center, img2img_floor, floor_depth, mask_floor, fx, fy)
    ceiling_pano, ceiling_pano_img = ceiling_to_pano(rot, pos2, center, img2img_ceiling, ceiling_depth, mask_ceiling, fx, fy)
    final_pano = integrate(inpainted_stitched_floor, floor_pano_img, ceiling_pano_img)
    final_pano.save(cfg['img_path']['pano_wall_2K'])
    