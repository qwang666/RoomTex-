import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector
import numpy as np
import torch
from utils.cameras.cameras import Cameras, CameraType


def capture_pano_depth(mesh, img_h, rot, pano_cam_center):
    c2w = torch.tensor([[[np.cos(rot), 0., -np.sin(rot), pano_cam_center[0]],
                         [0., 1., 0., pano_cam_center[1]],
                         [np.sin(rot), 0., np.cos(rot), pano_cam_center[2]]]])

    equirectangular_camera = Cameras(
        cx=float(img_h),
        cy=0.5 * float(img_h),
        fx=float(img_h),
        fy=float(img_h),
        width=2 * img_h,
        height=img_h,
        camera_to_worlds=c2w,
        camera_type=CameraType.EQUIRECTANGULAR,
    )
    camera_ray_bundle = equirectangular_camera.generate_rays(camera_indices=0)

    rays_o = camera_ray_bundle.origins  # [h, w, 3]
    rays_d = camera_ray_bundle.directions  # [h, w, 3]
    rays_o = rays_o.view(-1, 3)  # [h*w, 3]
    rays_d = rays_d.view(-1, 3)  # [h*w, 3]

    points, index_ray, _ = RayMeshIntersector(
        mesh).intersects_location(rays_o, rays_d, multiple_hits=False)

    coords = np.array(list(np.ndindex(img_h, img_h*2))).reshape(img_h,
                                                                img_h*2, -1).reshape(-1, 2)  # .transpose(1,0,2).reshape(-1,2)
    depth = trimesh.util.diagonal_dot(
        points - rays_o[0].numpy(), rays_d[index_ray].numpy())
    pixel_ray = coords[index_ray]
    # no depth value set 0.1
    depthmap = np.full([img_h, img_h*2], 0.1)
    depthmap[pixel_ray[:, 0], pixel_ray[:, 1]] = depth

    disp = 1/depthmap
    disp[disp == 10] = 0

    mask = depthmap != 0.1
    mask = np.stack([mask]*3, axis=2)*255

    return depthmap, disp, mask


def get_depth(pose, mesh, resolution, f=500.0):
    scale = resolution/1024.0
    # transform pose in different coordinate system
    depth_pose = pose.clone()
    depth_pose[:, :, 1] = -depth_pose[:, :, 1]
    depth_pose[:, :, 2] = -depth_pose[:, :, 2]

    c2w = depth_pose[0, :3, :]

    width_pers = int(1024*scale)
    height_pers = int(1024*scale)

    cx = (width_pers-1)/2
    cy = (height_pers-1)/2
    fx = f*scale
    fy = f*scale

    perspective_camera = Cameras(
        cx=cx,
        cy=cy,
        fx=fx,
        fy=fy,
        width=width_pers,
        height=height_pers,
        camera_to_worlds=c2w,
        camera_type=CameraType.PERSPECTIVE,
    )
    camera_ray_bundle = perspective_camera.generate_rays(camera_indices=0)

    rays_o = camera_ray_bundle.origins  # [h, w, 3]
    rays_d = camera_ray_bundle.directions  # [h, w, 3]
    rays_o = rays_o.view(-1, 3)  # [h*w, 3]
    rays_d = rays_d.view(-1, 3)  # [h*w, 3]

    points, index_ray, _ = RayMeshIntersector(
        mesh, False).intersects_location(rays_o, rays_d, multiple_hits=False)

    coords = np.array(list(np.ndindex(height_pers, width_pers))).reshape(
        height_pers, width_pers, -1).reshape(-1, 2)  # .transpose(1,0,2).reshape(-1,2)
    # depth = trimesh.util.diagonal_dot(points - rays_o[0].numpy(), rays_d[index_ray].numpy())
    c2w = torch.concat(
        (c2w, torch.tensor([0.0, 0.0, 0.0, 1.0]).reshape(1, 4)), dim=0)
    points = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
    depth = np.abs((torch.inverse(c2w).numpy() @ points.T)[2])
    # depth = (points @ rotation.numpy() + trans.numpy())[:,-1]
    pixel_ray = coords[index_ray]
    # 创建深度图矩阵，并进行对应赋值，没值的地方为nan，即空值
    depthmap_pers = np.full([height_pers, width_pers], 0.1)
    depthmap_pers[pixel_ray[:, 0], pixel_ray[:, 1]] = depth
    return depthmap_pers
