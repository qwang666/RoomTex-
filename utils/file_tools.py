
import os
import yaml


def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_cfg(path):
    with open(os.path.expanduser(path), "r") as config:
        cfg = yaml.safe_load(config)
    return cfg
