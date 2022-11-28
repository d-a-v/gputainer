import os
from pathlib import Path

# Folder of the dataset images, in which train_pbr folder is located
# GENERATED_DS_DIR = "blocks_dataset"
# GENERATED_DS_DIR = "blocks_multi"
GENERATED_DS_DIR = "tless"
# GENERATED_DS_DIR = "blocks_uni"

CUSTOM_DS_DIR = Path(os.environ["SCRATCH"]) / "/net/cubitus/projects/datasets4dl/users/emaitre/cosypose/data/scenes" / GENERATED_DS_DIR
print("path to scratch", Path(os.environ["SCRATCH"]))
print("custom ds dir = ", CUSTOM_DS_DIR)
DEST_PLY_DIR = CUSTOM_DS_DIR / "models"

# Folder of the URDF models
# CAD_DATASET_NAME = "cubes_dataset"
CAD_DATASET_NAME = "tless.cad"
# CAD_DATASET_NAME = "cubes_dataset_uni"
CAD_MODELS_DIR = Path("/net/cubitus/projects/datasets4dl/cosypose/local_data/urdfs/") / CAD_DATASET_NAME
