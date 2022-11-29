import pandas as pd
from pathlib import Path

from cosypose.config import ASSET_DIR
from cosypose.utils.logging import get_logger

from custom_training.custom_cosypose.datasets.custom_object_datasets import CustomObjectDataset
from custom_training.custom_cosypose.datasets.custom import CustomDataset
from cosypose.datasets.urdf_dataset import BOPUrdfDataset, OneUrdfDataset


logger = get_logger(__name__)


def make_scene_dataset(ds_name, n_frames=None, custom_ds_dir=None):
    if 'custom.' in ds_name:
        ds_dir = custom_ds_dir
        ds = CustomDataset(ds_dir, split='train_pbr')
    else:
        raise ValueError(ds_name)

    if n_frames is not None:
        ds.frame_index = ds.frame_index.iloc[:n_frames].reset_index(drop=True)
    ds.name = ds_name
    return ds


def make_object_dataset(ds_name, custom_ds_dir=None, custom_labels=None):
    ds = None
    if ds_name == 'custom':
        ds = CustomObjectDataset(Path(custom_ds_dir) / 'models', custom_labels)

    else:
        raise ValueError(ds_name)
    return ds


def make_urdf_dataset(ds_name, urdf_dataset_dir=None):
    if isinstance(ds_name, list):
        ds_index = []
        for ds_name_n in ds_name:
            dataset = make_urdf_dataset(ds_name_n)
            ds_index.append(dataset.index)
        dataset.index = pd.concat(ds_index, axis=0)
        return dataset

    # BOP
    if ds_name == 'own_dataset':
        ds = BOPUrdfDataset(urdf_dataset_dir)

    elif ds_name == 'camera':
        ds = OneUrdfDataset(ASSET_DIR / 'camera/model.urdf', 'camera')
    else:
        raise ValueError(ds_name)
    return ds
