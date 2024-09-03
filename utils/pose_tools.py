
import matplotlib as mpl
import torch
import trimesh
import cv2
import numpy as np
from utils.world_points import get_center_random_pos
import random


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


def rot_x(theta_range):
    rotation_x = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(theta_range), -np.sin(theta_range)],
        [0.0, np.sin(theta_range), np.cos(theta_range)],

    ]).float()
    return rotation_x


def rot_z(theta_range):
    rotation_z = torch.tensor([
        [np.cos(theta_range), -np.sin(theta_range), 0.0],
        [np.sin(theta_range), np.cos(theta_range), 0.0],
        [0.0, 0.0, 1.0],

    ]).float()
    return rotation_z


def rot_y(phi_range):
    rotation_y = torch.tensor([
        [np.cos(phi_range), 0.0, np.sin(phi_range)],
        [0.0, 1.0, 0.0],
        [-np.sin(phi_range), 0.0, np.cos(phi_range)],

    ]).float()
    return rotation_y


def rot_error(r_gt, r_est):
    dis = abs(np.arccos((np.trace(np.dot(np.linalg.inv(r_gt), r_est))-1)/2))

    return dis*180/np.pi


def get_four_pose(mesh, xyz_bound, scale=1, z_degree=4*np.pi/18):
    v = mesh.vertices
    # print((v[:,0].min()+v[:,0].max())/2)
    # print((v[:,1].min()+v[:,1].max())/2)
    # print((v[:,2].min()+v[:,2].max())/2)
    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         (v[:, 2].min()+v[:, 2].max())/2])]*4, dim=0)

    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max() -
                v[:, 1].min())**2 + (v[:, 2].max()-v[:, 2].min())**2)
    # print(r*scale)
    # 45 degree normal pose
    z_value = np.sin(z_degree)
    ang = np.pi/4
    position = torch.FloatTensor([
        [np.cos(ang), np.sin(ang), z_value],
        [-np.cos(ang), np.sin(ang), z_value],
        [-np.cos(ang), -np.sin(ang), z_value],
        [np.cos(ang), -np.sin(ang), z_value],
    ])

    position = position*r*scale+center
    position[:, 0] = torch.clamp(
        position[:, 0], min=xyz_bound[0, 0]+0.1, max=xyz_bound[0, 1]-0.1)
    position[:, 1] = torch.clamp(
        position[:, 1], min=xyz_bound[1, 0]+0.1, max=xyz_bound[1, 1]-0.1)
    if v[:, 2].max()+0.1*r*scale <= xyz_bound[2, 1]-0.1:
        position[:, 2] = torch.clamp(
            position[:, 2], min=v[:, 2].max()+0.15*r*scale, max=xyz_bound[2, 1]-0.1)
    else:
        position[:, 2] = torch.clamp(
            position[:, 2], min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)
    c2w = get_poses(position, center)

    return c2w


def get_four_position(rotation=np.pi/4, z_degree=4*np.pi/18):
    if z_degree == 1:
        z_degree = random.uniform(np.pi/6, np.pi/3)
    elif z_degree == -1:
        z_degree = -random.uniform(np.pi/6, np.pi/3)

    z_value = np.sin(z_degree)

    position = []
    for i in range(4):
        pos = [np.sin(i*2*np.pi/4+rotation),
               np.cos(i*2*np.pi/4+rotation), z_value]
        position.append(pos)
    position = torch.FloatTensor(position)

    return position


def get_refine_pose(mesh, z_value, xyz_bound, scale=1):
    v = mesh.vertices
    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 +
                (v[:, 1].max()-v[:, 1].min())**2)
    mesh_boundry = np.array([(v[:, 0].max()-v[:, 0].min()),
                            (v[:, 1].max()-v[:, 1].min()),
                            (v[:, 2].max()-v[:, 2].min()),
                             ])
    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         v[:, 2].min()])]*8, dim=0)
    posi_list = []
    for i in range(8):
        posi_list.append([np.sin(i*np.pi/4), np.cos(i*np.pi/4), 0])
    position = torch.FloatTensor(
        posi_list)*r*scale + torch.FloatTensor([0, 0, z_value*0.7+0.1])
    position = position+center
    position = np.array(position)
    center = np.array(center)
    xyz_bound = np.array(xyz_bound)
    a = (position[:, 0] < xyz_bound[0, 0]) + (position[:, 0] > xyz_bound[0, 1]) + (position[:, 1] < xyz_bound[1, 0]) + \
        (position[:, 1] > xyz_bound[1, 1]) + (position[:, 2] <
                                              xyz_bound[2, 0]) + (position[:, 2] > xyz_bound[2, 1])

    mask = a > 0

    position = np.delete(position, np.where(mask), axis=0)
    center = np.delete(center, np.where(mask), axis=0)

    position = torch.FloatTensor(position)
    center = torch.FloatTensor(center)

    c2w = get_poses(position, center)

    return c2w


def get_four_ador_pose(mesh, xyz_bound, scale=0.8, z_degree=np.pi/18):
    v = mesh.vertices
    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max() -
                v[:, 1].min())**2 + (v[:, 2].max()-v[:, 2].min())**2)
    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         (v[:, 2].min()+v[:, 2].max())/2])]*8, dim=0)
    mesh_boundry = np.array([(v[:, 0].max()-v[:, 0].min()),
                            (v[:, 1].max()-v[:, 1].min()),
                            (v[:, 2].max()-v[:, 2].min()),
                             ])
    position = torch.FloatTensor([
        [mesh_boundry[0]+0.15, mesh_boundry[1]+0.15, mesh_boundry[2]],
        [mesh_boundry[0]+0.15, -mesh_boundry[1]-0.15, mesh_boundry[2]],
        [mesh_boundry[0]+0.15, mesh_boundry[1]+0.15, -mesh_boundry[2]],
        [mesh_boundry[0]+0.15, -mesh_boundry[1]-0.15, -mesh_boundry[2]],
        [-mesh_boundry[0]-0.15, mesh_boundry[1]+0.15, mesh_boundry[2]],
        [-mesh_boundry[0]-0.15, -mesh_boundry[1]-0.15, mesh_boundry[2]],
        [-mesh_boundry[0]-0.15, mesh_boundry[1]+0.15, -mesh_boundry[2]],
        [-mesh_boundry[0]-0.15, -mesh_boundry[1]-0.15, -mesh_boundry[2]],
    ])
    position = position*scale+center
    position[:, 2] = torch.clamp(
        position[:, 2], min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)
    position = np.array(position)
    center = np.array(center)
    xyz_bound = np.array(xyz_bound)
    a = (position[:, 0] < xyz_bound[0, 0]) + (position[:, 0] > xyz_bound[0, 1]) + (position[:, 1] < xyz_bound[1, 0]) + \
        (position[:, 1] > xyz_bound[1, 1]) + (position[:, 2] <
                                              xyz_bound[2, 0]) + (position[:, 2] > xyz_bound[2, 1])

    mask = a > 0

    position = np.delete(position, np.where(mask), axis=0)
    center = np.delete(center, np.where(mask), axis=0)

    position = torch.FloatTensor(position)
    center = torch.FloatTensor(center)

    c2w = get_poses(position, center)

    return c2w


def get_four_up_down_pose(mesh, xyz_bound, scale=0.7, z_degree=4*np.pi/18):
    v = mesh.vertices
    # print((v[:,0].min()+v[:,0].max())/2)
    # print((v[:,1].min()+v[:,1].max())/2)
    # print((v[:,2].min()+v[:,2].max())/2)
    mesh_boundry = np.array([(v[:, 0].max()-v[:, 0].min()),
                            (v[:, 1].max()-v[:, 1].min()),
                            (v[:, 2].max()-v[:, 2].min()),
                             ])
    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         (v[:, 2].min()+v[:, 2].max())/2])]*4, dim=0)
    center_down = center.clone()

    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max() -
                v[:, 1].min())**2 + (v[:, 2].max()-v[:, 2].min())**2)
    # print(r*scale)
    # 45 degree normal pose
    # z_value = np.sin(z_degree)
    # z_value_down = -np.sin(z_degree)+0.1*r*scale
    # ang = np.pi/4
    # position = torch.FloatTensor([
    #     [np.cos(ang), np.sin(ang), z_value],
    #     [-np.cos(ang), np.sin(ang), z_value],
    #     [-np.cos(ang), -np.sin(ang), z_value],
    #     [np.cos(ang), -np.sin(ang), z_value],
    # ])
    # position_down = torch.FloatTensor([
    #     [np.cos(ang), np.sin(ang), z_value_down],
    #     [-np.cos(ang), np.sin(ang), z_value_down],
    #     [-np.cos(ang), -np.sin(ang), z_value_down],
    #     [np.cos(ang), -np.sin(ang), z_value_down],
    # ])
    position = torch.FloatTensor([
        [mesh_boundry[0]*scale+0.15, mesh_boundry[1]
            * scale+0.15, mesh_boundry[2]*scale],
        [mesh_boundry[0]*scale+0.15, -mesh_boundry[1]
            * scale-0.15, mesh_boundry[2]*scale],
        [-mesh_boundry[0]*scale-0.15, mesh_boundry[1]
            * scale+0.15, mesh_boundry[2]*scale],
        [-mesh_boundry[0]*scale-0.15, -mesh_boundry[1]
            * scale-0.15, mesh_boundry[2]*scale],

    ])
    position_down = torch.FloatTensor([
        [mesh_boundry[0]*scale+0.15, mesh_boundry[1]
            * scale+0.15, -mesh_boundry[2]*scale],
        [mesh_boundry[0]*scale+0.15, -mesh_boundry[1]
            * scale-0.15, -mesh_boundry[2]*scale],
        [-mesh_boundry[0]*scale-0.15, mesh_boundry[1]
            * scale+0.15, -mesh_boundry[2]*scale],
        [-mesh_boundry[0]*scale-0.15, -mesh_boundry[1]
            * scale-0.15, -mesh_boundry[2]*scale],

    ])
    position = position+center
    position_down = position_down+center_down

    position[:, 0] = torch.clamp(
        position[:, 0], min=xyz_bound[0, 0]+0.1, max=xyz_bound[0, 1]-0.1)
    position[:, 1] = torch.clamp(
        position[:, 1], min=xyz_bound[1, 0]+0.1, max=xyz_bound[1, 1]-0.1)
    position[:, 2] = torch.clamp(
        position[:, 2], min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)

    position_down[:, 0] = torch.clamp(
        position_down[:, 0], min=xyz_bound[0, 0]+0.1, max=xyz_bound[0, 1]-0.1)
    position_down[:, 1] = torch.clamp(
        position_down[:, 1], min=xyz_bound[1, 0]+0.1, max=xyz_bound[1, 1]-0.1)
    position_down[:, 2] = torch.clamp(
        position_down[:, 2], min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)

    # position, center = remove_redundant_position(position, center, r)
    # position_down, center_down = remove_redundant_position(position_down, center_down, r)
    position = np.array(position)
    center = np.array(center)
    dist_posi = np.linalg.norm(position-center, axis=1)

    position = np.delete(position, np.where(dist_posi < r*0.5), axis=0)
    center = np.delete(center, np.where(dist_posi < r*0.5), axis=0)

    position = torch.FloatTensor(position)
    center = torch.FloatTensor(center)

    position_down = np.array(position_down)
    center_down = np.array(center_down)
    dist_posi = np.linalg.norm(position_down-center_down, axis=1)

    position_down = np.delete(
        position_down, np.where(dist_posi < r*0.5), axis=0)
    center_down = np.delete(center_down, np.where(dist_posi < r*0.5), axis=0)
    position_down = torch.FloatTensor(position_down)
    center_down = torch.FloatTensor(center_down)

    c2w = get_poses(position, center)
    c2w_down = get_poses(position_down, center_down)

    return c2w, c2w_down


def clamp_position(position, xyz_bound, d=0.1):
    position[:, 0] = torch.clamp(
        position[:, 0], min=xyz_bound[0, 0]+d, max=xyz_bound[0, 1]-d)
    position[:, 1] = torch.clamp(
        position[:, 1], min=xyz_bound[1, 0]+d, max=xyz_bound[1, 1]-d)
    position[:, 2] = torch.clamp(
        position[:, 2], min=xyz_bound[2, 0]+d, max=xyz_bound[2, 1]-d/2)
    return position


def get_focus_center(mesh, xyz_bound, scale=0.9):
    c2w = []
    v = mesh.vertices
    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         (v[:, 2].min()+v[:, 2].max())/2])]*4, dim=0)
    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max() -
                v[:, 1].min())**2 + (v[:, 2].max()-v[:, 2].min())**2)
    a = torch.tensor(v[:, 0].max()-v[:, 0].min())
    b = torch.tensor(v[:, 1].max()-v[:, 1].min())
    focus_exit = False
    # print("center: ", center)
    if a > b:
        if a/b > 1.5:
            # print("a/b: ", a/b)
            focus_exit = True
            c = torch.sqrt(a**2-b**2)
            # print("c: ", c)
            c = a/6
            # print("c: ", c)
            focus_center_1 = torch.FloatTensor([c, 0, 0])+center
            focus_center_2 = torch.FloatTensor([-c, 0, 0])+center
            # print("focus_center_1: ", focus_center_1)
            # print("focus_center_2: ", focus_center_2)
    else:
        if b/a > 1.5:
            # print("b/a: ", b/a)
            focus_exit = True
            c = torch.sqrt(b**2-a**2)
            c = a/5
            focus_center_1 = torch.FloatTensor([0, c, 0])+center
            focus_center_2 = torch.FloatTensor([0, -c, 0])+center
            # print("focus_center_1: ", focus_center_1)
            # print("focus_center_2: ", focus_center_2)
    if focus_exit:
        r = np.sqrt(c**2+(b/2)**2+(v[:, 2].max()-v[:, 2].min())**2)
        position_up = get_four_position(rotation=0, z_degree=1)
        position_down = get_four_position(rotation=0, z_degree=-1)
        focus_center_down_1 = focus_center_1.clone()
        focus_center_down_2 = focus_center_2.clone()
        position_up_1 = position_up*r*scale+focus_center_1
        position_up_2 = position_up*r*scale+focus_center_2
        position_down_1 = position_down*r*scale+focus_center_down_1
        position_down_2 = position_down*r*scale+focus_center_down_2

        position_up_1 = clamp_position(position_up_1, xyz_bound, d=0.1)
        position_up_2 = clamp_position(position_up_2, xyz_bound, d=0.1)
        position_down_1 = clamp_position(position_down_1, xyz_bound, d=0.1)
        position_down_2 = clamp_position(position_down_2, xyz_bound, d=0.1)

        position_up_1, focus_center_1 = remove_redundant_position(
            position_up_1, focus_center_1, v, r)
        position_up_2, focus_center_2 = remove_redundant_position(
            position_up_2, focus_center_2, v, r)
        position_down_1, focus_center_down_1 = remove_redundant_position(
            position_down_1, focus_center_down_1, v, r)
        position_down_2, focus_center_down_2 = remove_redundant_position(
            position_down_2, focus_center_down_2, v, r)
        position = torch.concat(
            [position_up_1, position_up_2, position_down_1, position_down_2], dim=0)

        center = torch.concat(
            [focus_center_1, focus_center_2, focus_center_down_1, focus_center_down_2], dim=0)
        # print(center.shape)
        # print(position.shape)
        if len(center) > 0:
            c2w = get_poses(position, center)
        else:
            c2w = []
    else:
        position_up = get_four_position(rotation=0, z_degree=np.pi/4)
        position_down = get_four_position(rotation=0, z_degree=-np.pi/4)
        position_up = position_up*r*0.7+center
        position_down = position_down*r*0.7+center

        position_up = clamp_position(position_up, xyz_bound, d=0.1)
        position_down = clamp_position(position_down, xyz_bound, d=0.1)

        position_up, center_up = remove_redundant_position(
            position_up, center, v, r)
        position_down, center_down = remove_redundant_position(
            position_down, center, v, r)

        position = torch.concat([position_up, position_down], dim=0)
        center = torch.concat([center_up, center_down], dim=0)

        if len(center) > 0:
            c2w = get_poses(position, center)
        else:
            c2w = []
    return c2w


def get_n_pose(mesh, xyz_bound, scale=0.7, z_degree=7*np.pi/36):
    v = mesh.vertices

    r = np.sqrt((v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max() -
                v[:, 1].min())**2 + (v[:, 2].max()-v[:, 2].min())**2)

    xy2 = (v[:, 0].max()-v[:, 0].min())**2 + (v[:, 1].max()-v[:, 1].min())**2
    half_num = int(xy2)
    half_num = max(half_num, 4)
    # print("half_num: ", half_num)

    center = torch.stack([torch.FloatTensor(
        [(v[:, 0].min()+v[:, 0].max())/2,
         (v[:, 1].min()+v[:, 1].max())/2,
         (v[:, 2].min()+v[:, 2].max())/2])]*half_num*2, dim=0)

    # 45 degree normal pose
    z_value = np.sin(z_degree)*r*scale+center[0, 2]

    if (z_value+0.15*r*scale) <= (xyz_bound[2, 1]-0.1):
        z_value = torch.clamp(
            z_value, min=v[:, 2].max()+0.15*r*scale, max=xyz_bound[2, 1]-0.1)

    else:
        z_value = torch.clamp(
            z_value, min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)

    z_value_neg = -np.sin(z_degree)*r*scale+center[0, 2]
    if (z_value_neg-0.15*r*scale) >= (xyz_bound[2, 0]+0.1):
        z_value_neg = torch.clamp(
            z_value_neg, min=xyz_bound[2, 0]+0.1, max=v[:, 2].min()-0.15*r*scale)
    else:
        z_value_neg = torch.clamp(
            z_value_neg, min=xyz_bound[2, 0]+0.1, max=xyz_bound[2, 1]-0.1)

    position = []
    for i in range(half_num):
        pos = [np.sin(i*2*np.pi/half_num), np.cos(i*2*np.pi/half_num), z_value]
        pos_neg = [np.sin((i+0.5)*2*np.pi/half_num),
                   np.cos((i+0.5)*2*np.pi/half_num), z_value_neg]
        position.append(pos)
        position.append(pos_neg)

    position = torch.FloatTensor(position)

    position[:, :2] = position[:, :2]*r*scale+center[:, :2]
    position[:, 0] = torch.clamp(
        position[:, 0], min=xyz_bound[0, 0]+0.1, max=xyz_bound[0, 1]-0.1)
    position[:, 1] = torch.clamp(
        position[:, 1], min=xyz_bound[1, 0]+0.1, max=xyz_bound[1, 1]-0.1)
    position = np.array(position)
    center = np.array(center)
    dist_posi = np.linalg.norm(position-center, axis=1)
    # print(dist_posi)
    # print("R: ", r)
    position = np.delete(position, np.where(dist_posi < r*0.5), axis=0)

    center = np.delete(center, np.where(dist_posi < r*0.5), axis=0)

    position = torch.FloatTensor(position)
    center = torch.FloatTensor(center)
    c2w = get_poses(position, center)
    print("Throw away {} poses".format(len(dist_posi)-len(position)))
    print("Total Number of Views: ", len(c2w))
    return c2w


def remove_redundant_position(position, center, v, r):
    r = np.array(r)
    position = np.array(position)
    center = np.array(center)
    dist_posi = np.linalg.norm(position-center, axis=1)

    # position = np.delete(position, np.where(dist_posi<r*0.5), axis=0)
    a = (position[:, 0] > v[:, 0].min()-0.1*r) * (position[:, 0] < v[:, 0].max()+0.1*r) * (position[:, 1] > v[:, 1].min()-0.1*r) * \
        (position[:, 1] < v[:, 1].max()+0.1*r) * (position[:, 2] >
                                                  v[:, 2].min()-0.1*r) * (position[:, 2] < v[:, 2].max()+0.1*r)
    b = dist_posi < r*0.6
    mask = (a+b) > 0
    # print("a: ", a)
    position = np.delete(position, np.where(mask), axis=0)
    # center = np.delete(center, np.where(dist_posi<r*0.5), axis=0)
    center = np.delete(center, np.where(mask), axis=0)

    position = torch.FloatTensor(position)
    center = torch.FloatTensor(center)

    return position, center


def get_rot_pose(center, pos_num=4, scale=1, z_degree=4*np.pi/18):
    z_value = np.sin(z_degree)

    position = []
    for i in range(pos_num):
        pos = [np.sin(i*2*np.pi/pos_num), np.cos(i*2*np.pi/pos_num), z_value]
        position.append(pos)
    position = torch.FloatTensor(position)

    position = position*scale+center

    c2w = get_poses(position, center)

    return c2w


def get_rot_center_pose(thetas, phis, centers):
    position = torch.tensor([[0.0, 0.0, 0.0]])

    forward_vector = (position-centers)

    new_vector = get_center_random_pos(forward_vector[0], thetas, phis)
    new_position = new_vector+centers
    new_poses = get_poses(new_position, centers)

    return new_poses
