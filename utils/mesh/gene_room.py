import numpy as np
import random
import os
import yaml
import sys
sys.path.append(os.getcwd())
from utils.file_tools import check_path, load_cfg
from utils.wall_tools import buil_wall, get_wall_pts, buil_baseboard_wall, build_hole_wall
from utils.mesh_tools import boundary_mesh, inner_boundary_mesh
import trimesh
from utils.ceil_tools import get_style_ceil, get_cycle_ceil_pts, build_cycle_ceil, get_cycle_normal_style_ceil, build_cycle_normal_ceil
import argparse


def gene_base_ceil(layout_pts, room_height):
    v = layout_pts + np.array([0, 0, room_height])
    f = np.array([
        [0, 1, 2],
        [0, 3, 2],
    ])
    return trimesh.Trimesh(v, f)


def gene_floor(layout_pts):
    v = layout_pts.copy()
    f = np.array([
        [0, 1, 2],
        [0, 3, 2],
    ])
    return trimesh.Trimesh(v, f)


def gene_normal_style_ceil(layout_pts, room_height, room_center_bottom, total_height=0.1, total_width=0.2, layers=1, style="triangle"):
    ceil_outer_pts = layout_pts + np.array([0, 0, room_height])
    room_center_top = layout_pts + np.array([0, 0, room_height])

    ceil = get_style_ceil(ceil_outer_pts, total_height, total_width,
                          room_center_top, room_center_bottom, layers=layers, style=style)
    return ceil


def gene_cycle_style_ceil(layout_pts, room_height, room_center_bottom, width=0.1, gap=0.2, height=0.1, num=100, ntype=5):
    outer_pts, inner_pts = get_cycle_ceil_pts(room_height,
                                              layout_pts,
                                              width,
                                              gap,
                                              num,
                                              ntype)
    v, f = build_cycle_ceil(inner_pts,
                            outer_pts,
                            height,
                            room_center_bottom,
                            thick_out=False)
    return trimesh.Trimesh(v, f)


def gene_cycle_normal_style_ceil(layout_pts, room_height, room_center_bottom, width=0.1, height=0.1, num=100, ntype=5):
    outer_pts, inner_pts = get_cycle_normal_style_ceil(room_height,
                                                       layout_pts,
                                                       width,
                                                       num,
                                                       ntype)
    v, f = build_cycle_normal_ceil(inner_pts,
                                   outer_pts,
                                   height,
                                   room_center_bottom,
                                   thick_out=False)

    return trimesh.Trimesh(v, f)


def gene_room_wall(cfg):

    x_min, x_max, y_min, y_max = cfg['room']['x_min'], cfg['room']['x_max'], cfg['room']['y_min'], cfg['room']['y_max']
    layout_pts = np.array([[x_min, y_min, 0.0],
                           [x_min, y_max, 0.0],
                           [x_max, y_max, 0.0],
                           [x_max, y_min, 0.0]])
    room_center_bottom = (layout_pts[2] - layout_pts[0])/2 + layout_pts[0]
    wall_thick = cfg['room']['wall_thick']
    room_height = cfg['room']['room_height']
    door_height = [random.uniform(1.9, min(2.1, room_height*0.9))]
    num_window = sum(np.array(cfg['room']['wall_type']) == "window_wall")
    wind_h_1 = []  # window distance from floor
    wind_h_2 = []  # window height
    for w in range(num_window):
        wind_h_1.append(random.uniform(0.5, 1.0))
        wind_h_2.append(random.uniform(
            1.5, min(2.0, 0.9*room_height-wind_h_1[w])))

    total_wall_pts = get_wall_pts(cfg['room']['wall_type'], cfg['room']['wall_pts'],
                                  cfg['room']['room_height'], door_height, wind_h_1, wind_h_2)
    total_wall_mesh = []
    for i in range(len(cfg['room']['wall_type'])):
        if cfg['room']['wall_type'][i] == 'wall':
            if not (cfg['room']['baseboard']):
                wall = buil_wall(
                    total_wall_pts[i], wall_thick, room_center_bottom, cfg['room']['thick_out'])
            else:
                wall = buil_baseboard_wall(
                    total_wall_pts[i], wall_thick, room_center_bottom, cfg['room']['thick_out'])
        elif cfg['room']['wall_type'][i] == 'door':
            wall = build_hole_wall(total_wall_pts[i][0], total_wall_pts[i][1], wall_thick, room_center_bottom,
                                   thick_out=cfg['room']['thick_out'], baseboard=cfg['room']['baseboard'], door=True)
        elif cfg['room']['wall_type'][i] == 'window_wall':
            wall = build_hole_wall(total_wall_pts[i][0], total_wall_pts[i][1], wall_thick, room_center_bottom,
                                   thick_out=cfg['room']['thick_out'], baseboard=cfg['room']['baseboard'])
        total_wall_mesh.append(wall)
    total_wall_mesh = trimesh.util.concatenate(total_wall_mesh)

    return total_wall_mesh


def gene_room_ceil(cfg):
    x_min, x_max, y_min, y_max = cfg['room']['x_min'], cfg['room']['x_max'], cfg['room']['y_min'], cfg['room']['y_max']
    layout_pts = np.array([[x_min, y_min, 0.0],
                           [x_min, y_max, 0.0],
                           [x_max, y_max, 0.0],
                           [x_max, y_min, 0.0]])
    room_height = cfg['room']['room_height']
    room_center_bottom = (layout_pts[2] - layout_pts[0])/2 + layout_pts[0]
    if cfg['room']['ceil_type'] == "random":
        ceil_type = ["None", "normal", "cycle"]  # , "cycle_normal"
        random_ceil = ceil_type[random.randrange(len(ceil_type))]
    else:
        random_ceil = cfg['room']['ceil_type']
    print("ceil type:{}".format(random_ceil))

    layer = None
    cycle_gap = None
    random_pattern = None
    ceil_height = random.uniform(0.1, 0.2)
    ceil_width = random.uniform(0.1, 0.2)
    if random_ceil == "normal":
        layer = random.randrange(1, 3)
        if layer == 1:
            ceil_height = random.uniform(0.05, 0.12)
            ceil_width = random.uniform(0.05, 0.12)
            pattern_type = ["rectangle", "triangle"]
            random_pattern = pattern_type[random.randrange(len(pattern_type))]
        else:
            random_pattern = "rectangle"
    elif random_ceil == "cycle":
        # cycle_width = random.uniform(0.09, 0.12)
        cycle_gap = random.uniform(0.2, 0.4)
        # cycle_height = random.uniform(0.18, 0.22)
        cycle_num = 100
        pattern_type = [0.5, 0.7, 0, 1, 2, 3, 4, 5]
        random_pattern = pattern_type[random.randrange(len(pattern_type))]
    elif random_ceil == "cycle_normal":
        # ceil_width = random.uniform(0.1, 0.3)
        # ceil_height = random.uniform(0.18, 0.22)
        cycle_num = 100
        pattern_type = [1, 2, 3, 4, 5]
        random_pattern = pattern_type[random.randrange(len(pattern_type))]
    print("ceil type:{}, ceil_height:{}, ceil_width:{}, layer:{}, pattern_type:{}, cycle_gap:{}".format(random_ceil,
                                                                                                        ceil_height,
                                                                                                        ceil_width,
                                                                                                        layer,
                                                                                                        random_pattern,
                                                                                                        cycle_gap))
    b_ceil = gene_base_ceil(layout_pts, room_height)
    if random_ceil == "normal":
        s_ceil = gene_normal_style_ceil(layout_pts,
                                        room_height,
                                        room_center_bottom,
                                        total_height=ceil_height,
                                        total_width=ceil_width,
                                        layers=layer,
                                        style=random_pattern)
        b_ceil = trimesh.util.concatenate([b_ceil, s_ceil])
    elif random_ceil == "cycle":
        s_ceil = gene_cycle_style_ceil(layout_pts,
                                       room_height,
                                       room_center_bottom,
                                       width=ceil_width,
                                       gap=cycle_gap,
                                       height=ceil_height,
                                       num=cycle_num,
                                       ntype=random_pattern)
        b_ceil = trimesh.util.concatenate([b_ceil, s_ceil])
    elif random_ceil == "cycle_normal":
        s_ceil = gene_cycle_normal_style_ceil(layout_pts,
                                              room_height,
                                              room_center_bottom,
                                              width=ceil_width,
                                              height=ceil_height,
                                              num=cycle_num,
                                              ntype=random_pattern)
        b_ceil = trimesh.util.concatenate([b_ceil, s_ceil])
    return b_ceil, random_ceil, random_pattern, layer


def gene_empty_room(cfg):
    x_min, x_max, y_min, y_max = cfg['room']['x_min'], cfg['room']['x_max'], cfg['room']['y_min'], cfg['room']['y_max']
    layout_pts = np.array([[x_min, y_min, 0.0],
                           [x_min, y_max, 0.0],
                           [x_max, y_max, 0.0],
                           [x_max, y_min, 0.0]])
    wall = gene_room_wall(cfg, )
    ceil, random_ceil, random_pattern, layer = gene_room_ceil(cfg)
    floor = gene_floor(layout_pts)

    # total_mesh = trimesh.util.concatenate([wall, ceil, floor])
    total_mesh = ceil
    total_v = total_mesh.vertices
    new_v = total_v.copy()
    new_v[:, 2] = total_v[:, 1]
    new_v[:, 1] = total_v[:, 2]
    total_mesh = trimesh.Trimesh(vertices=new_v, faces=total_mesh.faces)
    return total_mesh, random_ceil, random_pattern, layer


def gene_boundry(cfg):
    x_max, y_max = cfg['room']['x_max'], cfg['room']['y_max']
    room_height = cfg['room']['room_height']
    wall_thick = cfg['room']['wall_thick']
    close_wall_mesh = boundary_mesh(x_max, y_max, room_height, wall_thick)
    return close_wall_mesh


def gene_inner_boundry(cfg):
    x_max, y_max = cfg['room']['x_max'], cfg['room']['y_max']
    room_height = cfg['room']['room_height']
    wall_thick = cfg['room']['wall_thick']
    close_wall_mesh = inner_boundary_mesh(x_max, y_max, room_height)
    return close_wall_mesh


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    obj_name = cfg['save_path'].split('/')[-1]
    save_path = 'demo/objects/{}/room'.format(obj_name)
    print(save_path)
    check_path(save_path)
    empty_room_mesh, random_ceil, random_pattern, layer = gene_empty_room(cfg)
    empty_room_mesh.export(save_path + "/{}_{}_layer{}_baseboard{}.obj".format(
        random_ceil, random_pattern, layer, cfg['room']['baseboard']))
    print(save_path+"/{}_{}_layer{}_baseboard{}.obj".format(random_ceil,
          random_pattern, layer, cfg['room']['baseboard']))
    boundry_mesh = gene_boundry(cfg)
    boundry_mesh.export(save_path + "/boundry.obj")
    print(save_path + "/boundry.obj")
