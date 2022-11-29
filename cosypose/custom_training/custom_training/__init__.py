import os
from pathlib import Path
import configparser

CONFIG_PATH = "./custom_training/.config"


def replace_env_variables(cfg):
    for key, val in cfg.items():
        if "$" in val:
            frags = val.split("/")
            for idx, frag in enumerate(frags):
                if "$" in frag:
                    name = frag.replace("$", "")
                    if name in os.environ.keys():
                        frags[idx] = os.environ[name]
                    else:
                        print(f"{frag} not in os environment variables.")
                    cfg[key] = "/".join(frags)
    return cfg


def read_config_file():
    DUMMY_TITLE = "DUMMY_CONFIG"

    with open(CONFIG_PATH, 'r') as f:
        config_string = f'[{DUMMY_TITLE}]\n' + f.read()
    config = configparser.ConfigParser()
    config.read_string(config_string)

    conf_dict = dict(config.items(DUMMY_TITLE))
    conf_dict = {key: val.replace("{", "").replace("}", "")
                 for key, val in conf_dict.items()}

    conf_dict = replace_env_variables(conf_dict)

    NB_SCENES_PER_BATCH = int(conf_dict["number_images"]) // (int(
        conf_dict["number_camera_poses_per_scene"]) * int(conf_dict["number_batches"]))
    conf_dict["number_scenes"] = NB_SCENES_PER_BATCH

    conf_dict["blender_install_path"] = str(
        Path(conf_dict["blender_path"]).parent)

    conf_dict["output_dir"] = str(
        Path(conf_dict["output_dir"]) / conf_dict["dataset_name"])
    conf_dict["logdir"] = str(
        Path(conf_dict["logdir"]) / f'{conf_dict["dataset_name"]}_blenderproc')

    return conf_dict
