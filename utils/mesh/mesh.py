import trimesh
from utils.mesh_tools import get_mesh_bound, load_obj_mesh
from utils.file_tools import check_path
from utils.mesh.gene_room import gene_empty_room, gene_boundry
from utils.file_tools import check_path, load_cfg
import argparse


def load_mesh(cfg):
    # load room mesh
    if cfg['exist_room']:
        mesh_empty_room = trimesh.load(cfg['room_mesh_path'])
        mesh_boundry = trimesh.load(cfg['boundry_mesh_path'])
    else:
        mesh_empty_room, random_ceil, random_pattern, layer = gene_empty_room(
            cfg)
        mesh_boundry = gene_boundry(cfg)
        obj_save_path = cfg['save_path']+"/obj_files/"
        check_path(obj_save_path)
        mesh_empty_room.export(obj_save_path+"/{}_{}_layer{}_baseboard{}.obj".format(
            random_ceil, random_pattern, layer, cfg['room']['baseboard']))
        mesh_boundry.export(obj_save_path+"/boundry.obj")
    mesh_obj_list = []
    for i in range(len(cfg['obj_id'])):
        mesh_obj = load_obj_mesh(cfg['obj_mesh_path'][i], cfg['obj_init_pos']
                                 [i], cfg['obj_init_scale'][i], cfg['obj_init_rot'][i])
        mesh_obj_list.append(mesh_obj)
    mesh_total_obj = trimesh.util.concatenate(mesh_obj_list)

    return mesh_empty_room, mesh_total_obj, mesh_obj_list, mesh_boundry


def load_adornment(cfg):
    mesh_adornment_list = []
    for i in range(len(cfg['adornment_id'])):
        mesh_obj = load_obj_mesh(cfg['adornment_mesh_path'][i], cfg['adornment_pos']
                                 [i], cfg['adornment_scale'][i], cfg['adornment_rot'][i])
        mesh_adornment_list.append(mesh_obj)
    mesh_adornment = trimesh.util.concatenate(mesh_adornment_list)
    return mesh_adornment, mesh_adornment_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    obj_name = cfg['save_path'].split('/')[-1]
    save_path = 'demo/{}'.format(obj_name)

    mesh_empty_room, mesh_total_obj, mesh_obj_list, mesh_boundry = load_mesh(
        cfg)
    mesh_empty_room.export(save_path + "/empty_room.obj")
    mesh_total_obj.export(save_path + "/total_obj.obj")
    mesh_room_boundry = trimesh.util.concatenate(
        [mesh_empty_room, mesh_boundry])
    mesh_room_boundry.export(save_path + "/boundry_empty_room.obj")
