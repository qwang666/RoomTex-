import trimesh
import torch
import numpy as np


def combine_two_mesh(v1, f1, v2, f2):
    v = np.concatenate([v1, v2], axis=0)
    f = np.concatenate([f1, f2 + v1.shape[0]], axis=0)
    return v, f


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


def rotate_mesh(v, ang_x, ang_y, ang_z):
    center = (v.max(axis=0) + v.min(axis=0)) / 2
    v = v - center
    v = rot_z(ang_z) @ rot_y(ang_y) @ rot_x(ang_x) @ v.T
    v = v.T

    return np.array(v)


def pose_normalize_mesh(v, origin, target_scale):
    # Compute center of bounding box)
    # center = v.mean(axis=0)
    center = (v.max(axis=0) + v.min(axis=0)) / 2
    v = v - center
    scale = np.max(np.linalg.norm(v, axis=1))
    v = (v / scale) * target_scale

    v = v + origin
    # v[:, 2] -= v[:, 2].min()
    return v


def load_room_mesh(path, pano_center):
    mesh = trimesh.load(path, force='mesh')
    v, f = mesh.vertices, mesh.faces
    v[:, 0] -= pano_center[0]
    v[:, 1] -= pano_center[1]
    v[:, 2] -= pano_center[2]

    new_v = v.copy()
    new_v[:, 1] = -v[:, 2]
    new_v[:, 2] = v[:, 1]
    mesh = trimesh.Trimesh(vertices=new_v, faces=f)

    return mesh


def load_obj_mesh(path, position, scale, ang):
    mesh = trimesh.load(path, force='mesh')
    ang_x, ang_y, ang_z = np.pi*ang[0], np.pi*ang[1], np.pi*ang[2]
    v, f = mesh.vertices, mesh.faces
    # rotate
    v = rotate_mesh(v, ang_x, ang_y, ang_z)

    v = pose_normalize_mesh(v, position, scale)

    # v[:, 0] -= pano_center[0]
    # v[:, 1] -= pano_center[1]
    # v[:, 2] -= pano_center[2]

    # new_v = v.copy()
    # new_v[:, 1] = -v[:, 2]
    # new_v[:, 2] = v[:, 1]

    # dd = bottom-new_v[:,2].min()
    # new_v[:,2] += dd

    mesh = trimesh.Trimesh(vertices=v, faces=f)
    return mesh


def get_mesh_center(mesh):
    v = mesh.vertices
    center = torch.FloatTensor([[(v[:, 0].min()+v[:, 0].max())/2,
                                 (v[:, 1].min()+v[:, 1].max())/2,
                                 (v[:, 2].min()+v[:, 2].max())/2]])
    return center


def get_mesh_bound(mesh, wall_thick=0.2):
    v = mesh.vertices
    bound = torch.FloatTensor([[v[:, 0].min()+wall_thick, v[:, 0].max()-wall_thick],
                               [v[:, 1].min()+wall_thick, v[:, 1].max()-wall_thick],
                               [v[:, 2].min(), v[:, 2].max()]])
    return bound


def boundary_mesh(x_max, y_max, room_height, wall_thick):
    mesh = trimesh.creation.box(
        extents=[x_max+2*wall_thick, room_height, y_max+2*wall_thick])
    v = mesh.vertices
    v[:, 0] += x_max/2
    v[:, 1] += room_height/2
    v[:, 2] += y_max/2
    return trimesh.Trimesh(vertices=v, faces=mesh.faces)


def inner_boundary_mesh(x_max, y_max, room_height):
    mesh = trimesh.creation.box(extents=[x_max, room_height, y_max])
    v = mesh.vertices
    v[:, 0] += x_max/2
    v[:, 1] += room_height/2
    v[:, 2] += y_max/2
    return trimesh.Trimesh(vertices=v, faces=mesh.faces)


def stick_mesh(mesh1, mesh2):
    v1, f1 = mesh1.vertices, mesh1.faces
    v2, f2 = mesh2.vertices, mesh2.faces
    v = np.concatenate([v1, v2], axis=0)
    f = np.concatenate([f1, f2 + v1.shape[0]], axis=0)
    mesh = trimesh.Trimesh(vertices=v, faces=f)
    return mesh
