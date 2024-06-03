import argparse
import os
import sys
sys.path.append(os.getcwd())
from gene_img.pano.prepare_pano import prepare_pano_depth
from scripts.prepare_pers import pre_pers_depth
from utils.file_tools import load_cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, help='config file, yaml')
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)

    prepare_pano_depth(cfg)
    pre_pers_depth(cfg)
