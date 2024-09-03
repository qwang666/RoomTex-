import numpy as np
import trimesh


def get_wall_pts(wall_type, wall_pts, room_height, door_height, wind_h_1, wind_h_2):
    '''
    input:
        np.array()
        wall_pts: wall, n*2*3, two points of each wall
        door_wall_pts: wall with a door, n*4*3, four points of each wall
                        (for example, left wall, left wall with a door, right wall, right wall with a door)
        wind_h_1: height of window from the bottom of the wall
        wind_h_2: height of window
        windows_wall_pts: wall with a windows, n*4*3
    return:
        inner and outer points of each wall
    '''
    total_wall_pts = []
    door_num = 0
    window_num = 0
    for i in range(len(wall_type)):
        if wall_type[i] == 'wall':
            pts = []
            pts = np.array([
                wall_pts[i][0],
                wall_pts[i][0],
                wall_pts[i][1],
                wall_pts[i][1]
            ])
            pts[1, 2] = pts[0, 2] + room_height
            pts[2, 2] = pts[3, 2] + room_height
            total_wall_pts.append(pts)
        elif wall_type[i] == 'door':
            pts = []
            pts = np.array([
                wall_pts[i][0],
                wall_pts[i][0],
                wall_pts[i][3],
                wall_pts[i][3]
            ])
            pts[1, 2] = pts[0, 2] + room_height
            pts[2, 2] = pts[3, 2] + room_height

            inner_pts = np.array([
                wall_pts[i][1],
                wall_pts[i][1],
                wall_pts[i][2],
                wall_pts[i][2]
            ])
            inner_pts[1, 2] = inner_pts[0, 2] + door_height[door_num]
            inner_pts[2, 2] = inner_pts[3, 2] + door_height[door_num]
            total_wall_pts.append([inner_pts, pts])
            door_num += 1
        elif wall_type[i] == 'window_wall':
            pts = []
            pts = np.array([
                wall_pts[i][0],
                wall_pts[i][0],
                wall_pts[i][3],
                wall_pts[i][3]
            ])
            pts[1, 2] = pts[0, 2] + room_height
            pts[2, 2] = pts[3, 2] + room_height
            inner_pts = np.array([
                wall_pts[i][1],
                wall_pts[i][1],
                wall_pts[i][2],
                wall_pts[i][2]
            ])
            inner_pts[0, 2] = inner_pts[0, 2] + wind_h_1[window_num]
            inner_pts[3, 2] = inner_pts[3, 2] + wind_h_1[window_num]
            inner_pts[1, 2] = inner_pts[0, 2] + wind_h_2[window_num]
            inner_pts[2, 2] = inner_pts[3, 2] + wind_h_2[window_num]
            total_wall_pts.append([inner_pts, pts])
            window_num += 1
    return total_wall_pts


def buil_wall(outer_pts, thick, room_center_bottom, thick_out=True):
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

    thick_outer_pts = outer_pts + thick_dir*thick*aa

    f = np.array([
        [0, 1, 2],
        [0, 3, 2],
        [0, 3, 4],
        [3, 7, 4],
        [2, 3, 6],
        [3, 7, 6],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 1],
        [4, 5, 1],
        [1, 5, 2],
        [5, 6, 2],
    ])
    v = np.concatenate([outer_pts, thick_outer_pts])

    wall = trimesh.Trimesh(vertices=v, faces=f)
    return wall


def buil_baseboard_wall(outer_pts, thick, room_center_bottom, thick_out=True):
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

    thick_outer_pts = outer_pts + thick_dir*thick*aa

    f = np.array([
        [0, 1, 2],
        [0, 3, 2],
        [0, 3, 4],
        [3, 7, 4],
        [2, 3, 6],
        [3, 7, 6],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 1],
        [4, 5, 1],
        [1, 5, 2],
        [5, 6, 2],
    ])
    v = np.concatenate([outer_pts, thick_outer_pts])

    baseboard_height = 0.08
    baseboard_pts = np.array([
        outer_pts[0],
        outer_pts[0] + np.array([0, 0, baseboard_height]),
        outer_pts[3] + np.array([0, 0, baseboard_height]),
        outer_pts[3],
    ])
    baseboard = buil_wall(baseboard_pts, 0.03,
                          room_center_bottom, thick_out=False)

    wall = trimesh.Trimesh(vertices=v, faces=f)

    return trimesh.util.concatenate([wall, baseboard])


def build_hole_wall(inner_pts, outer_pts, thick, room_center_bottom, thick_out=True, baseboard=False, door=False):
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
    wall = trimesh.Trimesh(vertices=v, faces=f)
    if baseboard:
        baseboard_height = 0.08
        if not door:
            baseboard_pts = np.array([
                outer_pts[0],
                outer_pts[0] + np.array([0, 0, baseboard_height]),
                outer_pts[3] + np.array([0, 0, baseboard_height]),
                outer_pts[3],
            ])
            baseboard = buil_wall(baseboard_pts, 0.03,
                                  room_center_bottom, thick_out=False)
        else:
            baseboard_pts_1 = np.array([
                outer_pts[0],
                outer_pts[0] + np.array([0, 0, baseboard_height]),
                inner_pts[0] + np.array([0, 0, baseboard_height]),
                inner_pts[0],
            ])
            baseboard_1 = buil_wall(
                baseboard_pts_1, 0.03, room_center_bottom, thick_out=False)
            baseboard_pts_2 = np.array([
                inner_pts[3],
                inner_pts[3] + np.array([0, 0, baseboard_height]),
                outer_pts[3] + np.array([0, 0, baseboard_height]),
                outer_pts[3],
            ])
            baseboard_2 = buil_wall(
                baseboard_pts_2, 0.03, room_center_bottom, thick_out=False)
            baseboard = trimesh.util.concatenate([baseboard_1, baseboard_2])
        wall = trimesh.util.concatenate([wall, baseboard])

    return wall
