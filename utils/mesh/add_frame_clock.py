import trimesh
from utils.mesh_tools import load_obj_mesh
from utils.file_tools import check_path, load_cfg
import argparse


def stick_adornment_mesh(cfg):
    # load room mesh
    room_adornment = []
    mesh_empty_room = trimesh.load(cfg['room_mesh_path'])
    room_adornment.append(mesh_empty_room)
    for i in range(len(cfg['adornment_id'])):
        mesh_obj = load_obj_mesh(cfg['adornment_mesh_path'][i], cfg['adornment_pos']
                                 [i], cfg['adornment_scale'][i], cfg['adornment_rot'][i])
        room_adornment.append(mesh_obj)
    mesh_room_adornment = trimesh.util.concatenate(room_adornment)

    return mesh_room_adornment


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    obj_name = cfg['save_path'].split('/')[-1]
    save_path = 'demo/{}'.format(obj_name)

    mesh_room_adornment = stick_adornment_mesh(cfg)
    mesh_room_adornment.export(save_path + "/adornment_empty_room.obj")
