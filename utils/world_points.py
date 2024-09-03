import numpy as np
import cv2
import torch


def pano_2_pers(pano_img, pano_disp, new_poses, w=1024, f=500.0, rot=0):
    new_w = w
    new_h = w

    height, width = pano_img.shape[:2]

    pixel_offset = 0.5
    # projection

    image_coords = torch.meshgrid(torch.arange(
        height), torch.arange(width), indexing="ij")
    image_coords = (torch.stack(image_coords, dim=-1) +
                    pixel_offset).float()  # stored as (y, x) coordinates

    s_x = image_coords[:, :, 1]*2*torch.pi/width - torch.pi
    s_y = -image_coords[:, :, 0]*torch.pi/height + torch.pi/2

    pano_depth = torch.from_numpy(1/pano_disp).float()
    w_x = pano_depth * torch.sin(s_x) * torch.cos(s_y)
    w_y = pano_depth * torch.cos(s_x) * torch.cos(s_y)
    w_z = pano_depth * torch.sin(s_y)

    pano_world_coords = torch.stack(
        (w_x.view(-1), w_y.view(-1), w_z.view(-1), torch.ones_like(w_x.view(-1))), dim=0)

    c2w_rot = torch.tensor([[[np.cos(rot), np.sin(rot), 0.0, 0.0],
                             [-np.sin(rot), np.cos(rot), 0.0, 0.0],
                             [0., 0., 1., 0],
                             [0.0000,  0.0000,  0.0000,  1.0000]
                             ]]).float()
    pano_world_coords = torch.inverse(c2w_rot) @ pano_world_coords

    camera_coords = torch.inverse(new_poses) @ pano_world_coords

    cx = (new_w-1)/2
    cy = (new_h-1)/2
    fx = f
    fy = f

    new_intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])

    image_coords = new_intr @ camera_coords[:, 0:3, :]
    mask = image_coords[:, 2] > 1e-3
    z = image_coords[:, 2]
    z[z < 1e-3] = 1e-3

    u = (image_coords[:, 0]/z).view(-1)
    v = (image_coords[:, 1]/z).view(-1)

    mask = (u >= 0) & (u < (new_w-0.5)) & (v >=
                                           0) & (v < (new_h-0.5)) & mask.view(-1)
    mask = mask.bool()
    total_img_warp = np.zeros((new_h, new_w, 3))

    u_view = torch.round(u[mask]).int().numpy()
    v_view = torch.round(v[mask]).int().numpy()

    total_masked_img = (pano_img.reshape(-1, 3))[mask]
    z_depth = (z.view(-1))[mask]
    img_room_warp_depth = np.ones((new_h, new_w))*100
    for i in range(v_view.shape[0]):
        if img_room_warp_depth[v_view[i], u_view[i]] > z_depth[i]:
            total_img_warp[v_view[i], u_view[i]] = total_masked_img[i]
            img_room_warp_depth[v_view[i], u_view[i]] = z_depth[i]
        # if img_room_warp_depth[v_view[i], u_view[i]] > img_warp_depth[v_view[i], u_view[i]]:
        #         total_img_warp[v_view[i], u_view[i]] = img_warp[v_view[i], u_view[i]]
        #         img_room_warp_depth[v_view[i], u_view[i]] = img_warp_depth[v_view[i], u_view[i]]
    # show_img(total_img_warp.astype(np.uint8))
    return total_img_warp.astype(np.uint8)


def interpolate_depth(depth, scale):
    '''
    depth: (h, w) numpy
    return: (scale*h, scale*w) numpy
    '''
    h, w = depth.shape
    depth = torch.nn.functional.interpolate(torch.tensor(
        depth[None, None, :, :]), scale_factor=4, mode='bilinear', align_corners=True)
    depth = depth.squeeze(0).squeeze(0).numpy()

    return depth


def get_pix_coords(height, width):
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32),
                               torch.arange(0, width, dtype=torch.float32)])
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)))
    return xyz  # [3, H*W]


def img2world(img, depth, mask, intr, c2w):
    '''
    input:
        img: tensor (h, w, 3)
        depth: tensor (h, w)
        mask: (h, w)
        intr: tensor (3, 3)
        c2w: tensor (1, 4, 4)
    output:
        world_coords: tensor: (1, 4, h*w)
    '''
    height, width = depth.shape
    image_coords = get_pix_coords(height, width)
    inv_intr = torch.inverse(intr).float()
    camera_coords = torch.from_numpy(depth).view(1, -1) * image_coords
    camera_coords = torch.matmul(inv_intr, camera_coords.float())  # [3, H*W].
    camera_coords = torch.cat(
        (camera_coords, torch.ones((1, height*width))), dim=0).float()

    world_coords = torch.matmul(c2w, camera_coords)

    return world_coords


def read_scale_img_depth_mask(img_path, depth_path, mask_path, scale, kernel=np.ones((7, 7), np.uint8), iter=2):
    '''
        erode mask

    input:
        mask: inpainting areas are true
    '''
    depth = np.load(depth_path)  # npy
    depth = interpolate_depth(depth, scale)
    height, width = depth.shape

    mask = cv2.imread(mask_path)  # png

    # show_img(mask)
    if iter != 0:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), 2)
        mask = cv2.erode(mask, np.ones((5, 5), np.uint8), 2)
    for i in range(iter):
        mask = cv2.erode(mask, kernel, 2)
#     mask2 = cv2.imread(mask2_path)
#     print(mask.shape, mask2.shape)
    total_mask = mask
    # show_img(mask)
    # show_img(mask2)
    total_mask = cv2.resize(total_mask, (width, height))
#     show_img(total_mask)
    total_mask = total_mask[:, :, -1]

    total_mask = total_mask.reshape(-1) > 125

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height))

    cx = (width-1)/2
    cy = (height-1)/2
    fx = 500*scale
    fy = 500*scale

    intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])

    return img, depth, total_mask, intr

# def read_imgx4_depth_mask(img_path, depth_path, mask_path, scale, kernel_size=5, iter_num=1, init=True, sd_mask_path=None):
#     '''
#         erode mask

#     input:
#         mask: inpainting areas are true
#     '''
#     depth = np.load(depth_path) #npy 4k
#     height, width = depth.shape

#     mask = cv2.imread(mask_path) #png 4k
#     kernel=np.ones((kernel_size, kernel_size), np.uint8)
#     if not init:
#         sd_mask = cv2.imread(sd_mask_path)
#         sd_mask = cv2.resize(sd_mask, (width, height))

#         mask = sd_mask-(255-mask)
#         # mask = cv2.erode(mask, kernel, iter_num)

#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (width, height))

#     cx=(width-1)/2
#     cy=(height-1)/2
#     fx=500*scale
#     fy=500*scale

#     intr = torch.tensor([
#         [fx, 0, cx],
#         [0, fy, cy],
#         [0, 0, 1.0]
#     ])

#     return img, depth, mask[:,:,-1].reshape(-1), intr


def get_lap_mask_depth(depth):
    laplacian = cv2.Laplacian(depth, cv2.CV_64F, ksize=5)
    laplacian = cv2.convertScaleAbs(laplacian)

    ret, binary = cv2.threshold(laplacian, 1.0, 255, 0)
    contours, hierarchy = cv2.findContours(
        binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt_depth_list = []

    for i in range(len(contours)):
        cnt = contours[i]
        cnt_depth = 0
        for k in range(len(cnt)):
            x, y = cnt[k][0]
            # print(depth[y, x]/40)
            a, b, c, d = x-1, x+1, y-1, y+1
            if a < 0:
                a = 0
            if b >= depth.shape[1]:
                b = depth.shape[1]-1
            if c < 0:
                c = 0
            if d >= depth.shape[0]:
                d = depth.shape[0]-1
            cnt_depth += (depth[y, x]+depth[y, b] +
                          depth[d, x]+depth[c, x]+depth[y, a])/5
            # depth[y, x] = 255
        cnt_depth = cnt_depth/len(cnt)
        cnt_depth_list.append(cnt_depth)

    mask = np.ones_like(depth)
    for i in range(len(contours)):
        cnt = contours[i]
        mask = cv2.drawContours(
            mask, [cnt], -1, 0, int(((cnt_depth_list[i]**2)*10))+1)
    return mask


# def get_shift_mask(img, depth):
#     edge = cv2.Canny(img[:, :, 0], 200, 800)

#     laplacian = cv2.Laplacian(depth, cv2.CV_64F, ksize=5)
#     laplacian = cv2.convertScaleAbs(laplacian)

#     _, binary = cv2.threshold(laplacian, 0.9, 255, 0)
#     whole_binary_mask = cv2.dilate(binary, np.ones((21, 21), np.uint8), iterations=2)

#     combine_mask = cv2.bitwise_or(edge, binary)
#     combine_mask = cv2.bitwise_and(combine_mask, whole_binary_mask)

#     shift_mask = cv2.dilate(combine_mask, np.ones((7, 7), np.uint8), iterations=3)
#     shift_mask = cv2.erode(shift_mask, np.ones((7, 7), np.uint8), iterations=1)
#     shift_mask = cv2.erode(shift_mask, np.ones((5, 5), np.uint8), iterations=1)
#     return shift_mask

def get_shift_mask(img, depth):
    edge = cv2.Canny(img[:, :, 0], 50, 250)

    laplacian = cv2.Laplacian(depth, cv2.CV_64F, ksize=5)
    laplacian = cv2.convertScaleAbs(laplacian)

    _, binary = cv2.threshold(laplacian, 0.9, 255, 0)
    whole_binary_mask = cv2.dilate(
        binary, np.ones((21, 21), np.uint8), iterations=2)

    combine_mask = cv2.bitwise_or(edge, binary)
    combine_mask = cv2.bitwise_and(combine_mask, whole_binary_mask)

    shift_mask = cv2.dilate(combine_mask, np.ones(
        (7, 7), np.uint8), iterations=3)
    shift_mask = cv2.erode(shift_mask, np.ones((7, 7), np.uint8), iterations=3)
    shift_mask = cv2.dilate(shift_mask, np.ones(
        (5, 5), np.uint8), iterations=1)
    return shift_mask


def read_imgx4_depth_mask(img_path, depth_path, mask_path, scale, kernel_size=5, iter_num=1, init=True, sd_mask_path=None, f=500.0):
    '''
        erode mask

    input:
        mask: inpainting areas are true
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    depth = np.load(depth_path)  # npy 4k
    height, width = depth.shape

    mask = cv2.imread(mask_path)  # png 4k
    kernel = np.ones((kernel_size, kernel_size), np.uint8)*255
    shift_mask = get_shift_mask(img, depth)
    if not init:
        sd_mask = cv2.imread(sd_mask_path)

        sd_mask = cv2.resize(sd_mask, (width, height))
        mask = cv2.erode(mask, kernel, iter_num)
        mask = cv2.erode(mask, kernel, iter_num)
        sd_mask = sd_mask[:, :, -1] > 0
        mask = mask[:, :, -1] > 0
        mask = (mask * sd_mask * (shift_mask == 0))
    else:
        mask = (mask[:, :, -1] > 0) * (shift_mask == 0)
    # mask = mask[:, :, -1] > 0
    # img = cv2.resize(img, (width, height))

    cx = (width-1)/2
    cy = (height-1)/2
    fx = f*scale
    fy = f*scale

    intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])

    return img, depth, mask.reshape(-1), intr


def safe_normalize(x, eps=1e-20):
    return x / torch.sqrt(torch.clamp(torch.sum(x * x, -1, keepdim=True), min=eps))


def get_poses(position, centers):
    forward_vector = safe_normalize(centers-position)
    up_vector = torch.FloatTensor([0, 0.0, 1.0]).unsqueeze(0)
    right_vector = safe_normalize(
        torch.cross(forward_vector, up_vector, dim=-1))
    up_vector = safe_normalize(torch.cross(
        right_vector, forward_vector, dim=-1))

    poses = torch.eye(4, dtype=torch.float).unsqueeze(
        0).repeat(position.shape[0], 1, 1)
    poses[:, :3, :3] = torch.stack(
        (right_vector, -up_vector, forward_vector), dim=-1)
    poses[:, :3, 3] = position
    return poses


def get_center_random_pos(forward_vector, theta_range, phi_range):
    '''
    theta_range: left-right
    phi_range: up-down
    both positive means left and bottom
    '''
    rotation_x = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_range), -np.sin(theta_range)],
        [0.0, np.sin(theta_range), np.cos(theta_range)],

    ]]).float()
    rotation_z = torch.tensor([[
        [np.cos(theta_range), -np.sin(theta_range), 0.0],
        [np.sin(theta_range), np.cos(theta_range), 0.0],
        [0.0, 0.0, 1.0],

    ]]).float()
    rotation_y = torch.tensor([[
        [np.cos(phi_range), 0.0, np.sin(phi_range)],
        [0.0, 1.0, 0.0],
        [-np.sin(phi_range), 0.0, np.cos(phi_range)],

    ]]).float()
    return rotation_z@rotation_y@forward_vector


def world2img(pose, total_img, total_world_coords, total_mask, new_w=1024, new_h=1024, f=500.0):
    '''
    total_mask: bool
    '''
    camera_coords = torch.inverse(pose) @ total_world_coords

    cx = (new_w-1)/2
    cy = (new_h-1)/2
    fx = f
    fy = f

    new_intr = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1.0]
    ])
    image_coords = new_intr @ camera_coords[:, 0:3, :]
    mask = image_coords[:, 2] > 1e-3
    z = image_coords[:, 2]
    z[z < 1e-3] = 1e-3

    u = (image_coords[:, 0]/z).view(-1)
    v = (image_coords[:, 1]/z).view(-1)

    mask = (u >= 0) & (u < (new_w-0.5)) & (v >= 0) & (v <
                                                      (new_h-0.5)) & mask.view(-1) & total_mask
    mask = mask.bool()
    img_warp = np.zeros((new_h, new_w, 3))

    u_view = torch.round(u[mask]).int().numpy()
    v_view = torch.round(v[mask]).int().numpy()

    total_masked_img = total_img[mask.view(-1)]
    z_depth = (z.view(-1))[mask.view(-1)]
    img_warp_depth = np.ones((new_h, new_w))*100

    for i in range(v_view.shape[0]):
        if img_warp_depth[v_view[i], u_view[i]] > z_depth[i]:
            img_warp[v_view[i], u_view[i]] = total_masked_img[i]
            img_warp_depth[v_view[i], u_view[i]] = z_depth[i]
    return img_warp.astype(np.uint8), img_warp_depth

# def get_cv_sd_mask(img_warp, mask_obj, kernel_size=3, itera=2):
#         mask_obj = cv2.dilate(mask_obj, np.ones((3, 3), np.uint8), iterations=1)
#         mask = img_warp.sum(-1) != 0
#         mask = (np.concatenate([mask[:,:,None], mask[:,:,None], mask[:,:,None]], axis=-1)*255).astype(np.uint8)

#         kernel=np.ones((kernel_size, kernel_size), np.uint8)
#         new_mask = cv2.dilate(mask, kernel, itera)
#         new_mask = cv2.erode(new_mask, kernel, itera)

#         cv_mask = ((new_mask - mask).astype(bool)*255).astype(np.uint8)
#         # sd_mask= ((mask_obj - new_mask).astype(bool)*255).astype(np.uint8)
#         return cv_mask #, sd_mask

# def get_cv_sd_mask(img_warp, mask_obj, kernel_size=3, itera=2):
#     # mask_obj = cv2.dilate(mask_obj, np.ones((3, 3), np.uint8), iterations=1)
#     mask = img_warp.sum(-1) == 0
#     mask = (np.concatenate([mask[:,:,None], mask[:,:,None], mask[:,:,None]], axis=-1)*255).astype(np.uint8)

#     cv_mask = (mask*mask_obj).astype(np.uint8)*255
#     kernel=np.ones((kernel_size, kernel_size), np.uint8)
#     new_mask = cv2.dilate((255-mask), kernel, itera)
#     new_mask = cv2.erode(new_mask, kernel, itera)
#     new_mask = cv2.erode(new_mask, kernel, itera)
#     new_mask = cv2.dilate(new_mask, kernel, itera)

#     sd_mask = ((255-new_mask).astype(np.uint8)*255*mask_obj)

#     return cv_mask, sd_mask


def get_lap_mask(depth):
    laplacian = cv2.Laplacian(depth, cv2.CV_64F, ksize=5)
    laplacian = cv2.convertScaleAbs(laplacian)

    ret, binary = cv2.threshold(laplacian, 0.99, 255, 0)
    return binary


def get_cv_sd_mask(img_warp, depth_gt, mask_obj, kernel_size=3, itera=2):
    # mask_obj = cv2.dilate(mask_obj, np.ones((3, 3), np.uint8), iterations=1)
    mask = img_warp.sum(-1) == 0
    mask = (np.concatenate([mask[:, :, None], mask[:, :, None],
            mask[:, :, None]], axis=-1)*255).astype(np.uint8)
    mask_obj = mask_obj > 0
    cv_mask = (mask*mask_obj).astype(np.uint8)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    new_mask = cv2.dilate((255-mask), kernel, itera)
    new_mask = cv2.erode(new_mask, kernel, itera)
    new_mask = cv2.erode(new_mask, kernel, itera)
    new_mask = cv2.dilate(new_mask, kernel, itera)

    lap_mask = get_lap_mask(depth_gt)
    lap_mask = np.stack([lap_mask, lap_mask, lap_mask], axis=-1)

    sd_mask = ((255-new_mask).astype(np.uint8))

    sd_mask = cv2.dilate(sd_mask, np.ones((3, 3), np.uint8), iterations=1)

    lap_mask = cv2.bitwise_and(lap_mask, cv_mask)
    sd_mask = cv2.bitwise_or(lap_mask, sd_mask)
    sd_mask = (sd_mask*mask_obj)

    erode_img = cv2.erode(img_warp, np.ones((7, 7), np.uint8), iterations=2)
    sd_fill_mask = erode_img == 0
    sd_fill = cv2.inpaint(
        erode_img, 255*sd_fill_mask[:, :, 0].astype(np.uint8), 3, cv2.INPAINT_TELEA)

    return cv_mask, sd_mask, sd_fill

# def get_combine_img(pano_img, inpaint_warp_img, mask_obj):
#     mask = inpaint_warp_img.sum(-1) != 0
#     mask = np.stack((mask, mask, mask), axis=-1)
#     mask = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2)
#     mask = cv2.subtract(mask.astype(np.uint8), (mask_obj == 0).astype(np.uint8))

#     mask_bed = mask[:,:,0]!=0
#     mask_room = mask[:,:,0]==0
#     combine_render = cv2.bitwise_and(pano_img, pano_img, mask = mask_room.astype(np.uint8))
#     combine_pers = cv2.bitwise_and(inpaint_warp_img,inpaint_warp_img,mask = mask_bed.astype(np.uint8))
#     combine_img = cv2.add(combine_render,combine_pers)

#     # sd_mask = cv2.subtract((mask==0).astype(np.uint8), (mask_obj == 0).astype(np.uint8))

#     return combine_render, combine_pers, combine_img #, sd_mask*255


def get_combine_img(pano_img, inpaint_warp_img, sd_mask, mask_obj, sd_fill):
    mask = cv2.subtract((255-sd_mask.astype(np.uint8)),
                        (mask_obj != 0).astype(np.uint8))

    mask = (255-sd_mask.astype(np.uint8))*(mask_obj != 0)
    sd_fill_mask = (sd_mask.astype(np.uint8))*(mask_obj != 0)

    mask_bed = mask[:, :, 0] != 0
    mask_room = mask_obj[:, :, 0] == 0
    mask_fill = sd_fill_mask[:, :, 0] != 0
    combine_render = cv2.bitwise_and(
        pano_img, pano_img, mask=mask_room.astype(np.uint8))
    combine_pers = cv2.bitwise_and(
        inpaint_warp_img, inpaint_warp_img, mask=mask_bed.astype(np.uint8))
    sd_fill = cv2.bitwise_and(
        sd_fill, sd_fill, mask=mask_fill.astype(np.uint8))
    combine_img = cv2.add(combine_render, combine_pers)
    combine_img = cv2.add(combine_img, sd_fill)

    return combine_render, combine_pers, combine_img
