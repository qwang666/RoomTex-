import numpy as np
from utils.wall_tools import build_hole_wall
import random
import trimesh


def get_ceil_inner_pts(inner_thick, ceil_outer_pts, room_center_top):

    ceil_inner_pts = ceil_outer_pts.copy()
    for i in range(4):
        direction = room_center_top - ceil_outer_pts[i]

        if (direction*np.array([inner_thick, 0, 0])).sum() > 0:
            ceil_inner_pts[i, 0] = ceil_inner_pts[i, 0] + inner_thick
        else:
            ceil_inner_pts[i, 0] = ceil_inner_pts[i, 0] - inner_thick
        if (direction*np.array([0, inner_thick, 0])).sum() > 0:
            ceil_inner_pts[i, 1] = ceil_inner_pts[i, 1] + inner_thick
        else:
            ceil_inner_pts[i, 1] = ceil_inner_pts[i, 1] - inner_thick

    return ceil_inner_pts


def get_cycle_ceil_pts(room_height, layout_pts, width, gap, num, ntype):
    '''
    assume the room layout is a rectangle

    layout_pts: np.array([lb, lt, rt, rb]) lb (x, y, z)
    '''

    layout_width = np.abs(layout_pts[0][0] - layout_pts[2][0]) - 2*gap
    layout_height = np.abs(layout_pts[0][1] - layout_pts[2][1]) - 2*gap
    if ntype == 0:
        points = np.array([
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0],
        ])
    # get_pattern
    else:
        delta = 2/(2*num)
        points = np.ones(((2*num+1), 3), dtype=np.float64)
        for i in range(2*num+1):
            points[i, 0] = -1+delta*i

            points[i, 1] = (1-(np.abs(-1+delta*i))**ntype)**(1./ntype)

        symmetric_points = points.copy()
        symmetric_points[:, 0] = -1*points[:, 0]
        symmetric_points[:, 1] = -1*points[:, 1]
        points = np.concatenate([points, symmetric_points])
    outer_pts = points.copy()
    outer_pts[:, 0] = points[:, 0]*layout_width / \
        2 + (layout_pts[2][0] - layout_pts[0][0])/2
    outer_pts[:, 1] = points[:, 1]*layout_height / \
        2 + (layout_pts[2][1] - layout_pts[0][1])/2
    outer_pts[:, 2] = points[:, 2]*room_height + layout_pts[2][2]

    inner_pts = points.copy()
    inner_pts[:, 0] = points[:, 0] * \
        (layout_width-2*width)/2 + (layout_pts[2][0] - layout_pts[0][0])/2
    inner_pts[:, 1] = points[:, 1] * \
        (layout_height-2*width)/2 + (layout_pts[2][1] - layout_pts[0][1])/2
    inner_pts[:, 2] = points[:, 2]*room_height + layout_pts[2][2]

    return outer_pts, inner_pts


def get_cycle_normal_style_ceil(room_height, layout_pts, width, num, ntype):
    '''
    assume the room layout is a rectangle

    layout_pts: np.array([lb, lt, rt, rb]) lb (x, y, z)
    '''

    layout_width = np.abs(layout_pts[0][0] - layout_pts[2][0])
    layout_height = np.abs(layout_pts[0][1] - layout_pts[2][1])

    outer_pts = np.array([
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, -1.0, 1.0],
    ])
    # get_pattern

    delta = 2/(2*num)
    outer_points = np.ones(((2*num+1), 3), dtype=np.float64)
    points = np.ones(((2*num+1), 3), dtype=np.float64)
    for i in range(2*num+1):
        points[i, 0] = -1+delta*i

        points[i, 1] = (1-(np.abs(-1+delta*i))**ntype)**(1./ntype)

        if i < (num/2):
            outer_points[i, 0] = -1
            outer_points[i, 1] = (1-(np.abs(-1+delta*i))**ntype)**(1./ntype)
        elif i == (num/2):
            outer_points[i, 0] = -1
            outer_points[i, 1] = 1
        elif (num/2) < i < (3*num/2):
            outer_points[i, 0] = -1+delta*i
            outer_points[i, 1] = 1
        elif i == (3*num/2):
            outer_points[i, 0] = 1
            outer_points[i, 1] = 1
        elif i > (3*num/2):
            outer_points[i, 0] = 1
            outer_points[i, 1] = (1-(np.abs(-1+delta*i))**ntype)**(1./ntype)

    symmetric_points = points.copy()
    symmetric_points[:, 0] = -1*points[:, 0]
    symmetric_points[:, 1] = -1*points[:, 1]
    points = np.concatenate([points, symmetric_points])

    symmetric_outer_points = outer_points.copy()
    symmetric_outer_points[:, 0] = -1*outer_points[:, 0]
    symmetric_outer_points[:, 1] = -1*outer_points[:, 1]
    outer_points = np.concatenate([outer_points, symmetric_outer_points])

    outer_pts = outer_points.copy()
    outer_pts[:, 0] = outer_points[:, 0]*layout_width / \
        2 + (layout_pts[2][0] - layout_pts[0][0])/2
    outer_pts[:, 1] = outer_points[:, 1]*layout_height / \
        2 + (layout_pts[2][1] - layout_pts[0][1])/2
    outer_pts[:, 2] = outer_points[:, 2]*room_height + layout_pts[2][2]

    inner_pts = points.copy()
    inner_pts[:, 0] = points[:, 0] * \
        (layout_width-2*width)/2 + (layout_pts[2][0] - layout_pts[0][0])/2
    inner_pts[:, 1] = points[:, 1] * \
        (layout_height-2*width)/2 + (layout_pts[2][1] - layout_pts[0][1])/2
    inner_pts[:, 2] = points[:, 2]*room_height + layout_pts[2][2]

    return outer_pts, inner_pts


def get_style_ceil(ceil_outer_pts, total_height, total_width, room_center_top, room_center_bottom, layers=1, style="rectangle"):
    if layers == 1:
        ceil_inner_pts = get_ceil_inner_pts(
            total_width, ceil_outer_pts, room_center_top)
        if style == "rectangle":
            ceil = build_hole_wall(
                ceil_inner_pts, ceil_outer_pts, total_height, room_center_bottom, thick_out=False)
            print("rectangle ceil style, layer = 1")
        if style == "triangle":
            ceil = build_triangle_ceil(
                ceil_inner_pts, ceil_outer_pts, total_height, room_center_bottom, thick_out=False)
            print("rectangle ceil style, layer = 1")
    elif layers == 2:
        height = random.uniform(0.2, 0.8)
        height = (int(height * 1000) / 1000)*total_height
        width = random.uniform(0.1, 0.7)
        width = (int(width * 1000) / 1000)*total_width
        ceil_inner_pts_top = get_ceil_inner_pts(
            total_width, ceil_outer_pts, room_center_top)
        ceil_outer_pts_bottom = ceil_outer_pts.copy()
        ceil_outer_pts_bottom[:, 2] = ceil_outer_pts_bottom[:, 2] - height
        ceil_inner_pts_bottom = get_ceil_inner_pts(
            width, ceil_outer_pts_bottom, room_center_top)
        if style == "rectangle":
            ceil_top = build_hole_wall(
                ceil_inner_pts_top, ceil_outer_pts, height, room_center_bottom, thick_out=False)
            ceil_bottom = build_hole_wall(
                ceil_inner_pts_bottom, ceil_outer_pts_bottom, total_height-height, room_center_bottom, thick_out=False)
            ceil = trimesh.util.concatenate(ceil_top, ceil_bottom)
            print("{} ceil style, layer = {}, top_height = {}, top_width = {}, bottom_height = {}, bottom__width = {}".format(style,
                                                                                                                              layers,
                                                                                                                              height,
                                                                                                                              total_width,
                                                                                                                              total_height-height,
                                                                                                                              width))
    elif layers == 3:
        height1 = random.uniform(0.1, 0.45)
        height1 = (int(height1 * 1000) / 1000)*total_height
        height2 = random.uniform(0.5, 0.9)
        height2 = (int(height2 * 1000) / 1000)*total_height

        width1 = random.uniform(0.1, 0.45)
        width1 = (int(width1 * 1000) / 1000)*total_width
        width2 = random.uniform(0.5, 0.9)
        width2 = (int(width2 * 1000) / 1000)*total_width

        ceil_inner_pts_top = get_ceil_inner_pts(
            total_width, ceil_outer_pts, room_center_top)
        ceil_outer_pts_middle = ceil_outer_pts.copy()
        ceil_outer_pts_middle[:, 2] = ceil_outer_pts_middle[:, 2] - height1

        ceil_inner_pts_middle = get_ceil_inner_pts(
            width2, ceil_outer_pts_middle, room_center_top-np.array([0, 0, height1]))
        ceil_outer_pts_bottom = ceil_outer_pts.copy()
        ceil_outer_pts_bottom[:, 2] = ceil_outer_pts_bottom[:, 2] - height2
        ceil_inner_pts_bottom = get_ceil_inner_pts(
            width1, ceil_outer_pts_bottom, room_center_top-np.array([0, 0, height2]))

        if style == "rectangle":
            ceil_top = build_hole_wall(
                ceil_inner_pts_top, ceil_outer_pts, height1, room_center_bottom, thick_out=False)
            ceil_middle = build_hole_wall(
                ceil_inner_pts_middle, ceil_outer_pts_middle, height2-height1, room_center_bottom, thick_out=False)
            ceil_bottom = build_hole_wall(
                ceil_inner_pts_bottom, ceil_outer_pts_bottom, total_height-height2, room_center_bottom, thick_out=False)
            ceil = trimesh.util.concatenate(
                [ceil_top, ceil_middle, ceil_bottom])
            print("{} ceil style, layer = {}, top_height = {}, top_width = {}, middle_height = {}, middle_width = {}, bottom_height = {}, bottom__width = {}".format(style,
                                                                                                                                                                     layers,
                                                                                                                                                                     height1,
                                                                                                                                                                     total_width,
                                                                                                                                                                     height2-height1,
                                                                                                                                                                     width2,
                                                                                                                                                                     total_height-height2,
                                                                                                                                                                     width1))

    return ceil


def build_triangle_ceil(inner_pts, outer_pts, thick, room_center_bottom, thick_out=False):
    """
    pts : np.array([lb, lt, rt, rb])
    """
    f = []
    h_direct = outer_pts[0]-outer_pts[1]
    v_direct = outer_pts[0]-outer_pts[3]

    thick_dir = np.cross(v_direct, h_direct)

    thick_dir = thick_dir/np.linalg.norm(thick_dir)

    room_dir = room_center_bottom-outer_pts[0]
    # in_out room direction
    aa = (room_dir*thick_dir).sum()
    if thick_out:
        aa = -1*np.sign(aa)
    # thick_inner_pts = inner_pts + thick_dir*thick*aa
    thick_outer_pts = outer_pts + thick_dir*thick*aa

    num_inner = len(inner_pts)
    num_outer = len(outer_pts)
    for i in range(num_inner):
        a = i+1
        if a > num_inner-1:
            a = 0
        # inner
        f.append([a, i, i+num_inner+num_outer])
        f.append([a+num_inner+num_outer, a, i+num_inner+num_outer])
        # wall
        # f.append([a+num_inner+num_outer, i+num_inner+num_outer, a+2*num_inner+num_outer])
        # f.append([a+2*num_inner+num_outer, i+num_inner+num_outer, i+2*num_inner+num_outer])
        # outer
        f.append([a+num_inner+num_outer, i+num_inner+num_outer, a+num_inner])
        f.append([a+num_inner, i+num_inner+num_outer, i+num_inner])
        # wall
        f.append([a+num_inner, i+num_inner, a])
        f.append([a, i+num_inner, i])
    v = np.concatenate([inner_pts, outer_pts, thick_outer_pts])
    f = np.array(f)
    return trimesh.Trimesh(v, f)


def build_cycle_ceil(inner_pts, outer_pts, thick, room_center_bottom, thick_out=True):
    f = []
    h_direct = outer_pts[0]-outer_pts[1]
    v_direct = outer_pts[0]-outer_pts[3]

    thick_dir = np.cross(v_direct, h_direct)

    thick_dir = thick_dir/np.linalg.norm(thick_dir)

    room_dir = room_center_bottom-outer_pts[0]
    # in_out room direction
    aa = (room_dir*thick_dir).sum()
    aa = np.sign(aa)
    if thick_out:
        aa = -1*aa

    thick_inner_pts = inner_pts + thick_dir*thick*aa
    thick_outer_pts = outer_pts + thick_dir*thick*aa

    num_inner = len(inner_pts)
    num_outer = len(outer_pts)
    for i in range(num_inner):
        a = i+1
        if a > num_inner-1:
            a = 0
        # inner
        f.append([a, i, i+num_inner+num_outer])
        f.append([a+num_inner+num_outer, a, i+num_inner+num_outer])
        # wall
        f.append([a+num_inner+num_outer, i+num_inner +
                 num_outer, a+2*num_inner+num_outer])
        f.append([a+2*num_inner+num_outer, i+num_inner +
                 num_outer, i+2*num_inner+num_outer])
        # outer
        f.append([a+2*num_inner+num_outer, i+2 *
                 num_inner+num_outer, a+num_inner])
        f.append([a+num_inner, i+2*num_inner+num_outer, i+num_inner])
        # wall
        f.append([a+num_inner, i+num_inner, a])
        f.append([a, i+num_inner, i])
    v = np.concatenate(
        [inner_pts, outer_pts, thick_inner_pts, thick_outer_pts])
    f = np.array(f)
    return v, f


def build_cycle_normal_ceil(inner_pts, outer_pts, thick, room_center_bottom, thick_out=True):
    f = []
    num_pts = len(outer_pts)
    h_direct = outer_pts[0]-outer_pts[1]
    v_direct = outer_pts[0]-outer_pts[int(num_pts/2)]

    thick_dir = np.cross(v_direct, h_direct)

    thick_dir = thick_dir/np.linalg.norm(thick_dir)

    room_dir = room_center_bottom-outer_pts[0]
    # in_out room direction
    aa = (room_dir*thick_dir).sum()
    aa = np.sign(aa)
    if thick_out:
        aa = -1*aa

    thick_inner_pts = inner_pts + thick_dir*thick*aa
    thick_outer_pts = outer_pts + thick_dir*thick*aa

    num_inner = len(inner_pts)
    num_outer = len(outer_pts)
    for i in range(num_inner):
        a = i+1
        if a > num_inner-1:
            a = 0
        # inner
        f.append([a, i, i+num_inner+num_outer])
        f.append([a+num_inner+num_outer, a, i+num_inner+num_outer])
        # wall
        f.append([a+num_inner+num_outer, i+num_inner +
                 num_outer, a+2*num_inner+num_outer])
        f.append([a+2*num_inner+num_outer, i+num_inner +
                 num_outer, i+2*num_inner+num_outer])
        # outer
        f.append([a+2*num_inner+num_outer, i+2 *
                 num_inner+num_outer, a+num_inner])
        f.append([a+num_inner, i+2*num_inner+num_outer, i+num_inner])
        # wall
        f.append([a+num_inner, i+num_inner, a])
        f.append([a, i+num_inner, i])
    v = np.concatenate(
        [inner_pts, outer_pts, thick_inner_pts, thick_outer_pts])
    f = np.array(f)
    return v, f
