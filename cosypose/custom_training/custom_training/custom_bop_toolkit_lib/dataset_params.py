# Adapted from https://github.com/thodan/bop_toolkit/blob/master/bop_toolkit_lib/dataset_params.py

"""Parameters of the dataset."""

import glob
import os
from os.path import join


def get_model_params(datasets_path, dataset_name, model_type=None, custom_nb_objects=None, custom_symmetric_obj_ids=None):
    """Returns parameters of object models for the specified dataset.

    :param datasets_path: Path to a folder with datasets.
    :param dataset_name: Name of the dataset for which to return the parameters.
    :param model_type: Type of object models.
    :return: Dictionary with object model parameters for the specified dataset.
    """
    # Object ID's.
    obj_ids = list(range(custom_nb_objects))

    # ID's of objects with ambiguous views evaluated using the ADI pose error
    # function (the others are evaluated using ADD). See Hodan et al. (ECCVW'16).
    symmetric_obj_ids = custom_symmetric_obj_ids

    # # T-LESS includes two types of object models, CAD and reconstructed.
    # # Use the CAD models as default.
    # if dataset_name == 'tless' and model_type is None:
    #     model_type = 'cad'

    # Name of the folder with object models.
    models_folder_name = 'models'
    if model_type is not None:
        models_folder_name += '_' + model_type

    # Path to the folder with object models.
    models_path = join(datasets_path, dataset_name, models_folder_name)

    model_tpath = join(models_path, 'categories')

    p = {
        # ID's of all objects included in the dataset.
        'obj_ids': obj_ids,

        # ID's of objects with symmetries.
        'symmetric_obj_ids': symmetric_obj_ids,

        # Path template to an object model file.
        'model_tpath': model_tpath,

        # Path to a file with meta information about the object models.
        'models_info_path': join(models_path, 'models_info.json')
    }

    return p


def get_split_params(datasets_path, dataset_name, split, split_type=None):
    """Returns parameters (camera params, paths etc.) for the specified dataset.

    :param datasets_path: Path to a folder with datasets.
    :param dataset_name: Name of the dataset for which to return the parameters.
    :param split: Name of the dataset split ('train', 'val', 'test').
    :param split_type: Name of the split type (e.g. for T-LESS, possible types of
      the 'train' split are: 'primesense', 'render_reconst').
    :return: Dictionary with parameters for the specified dataset split.
    """
    p = {
        'name': dataset_name,
        'split': split,
        'split_type': split_type,
        'base_path': join(datasets_path, dataset_name),

        'depth_range': None,
        'azimuth_range': None,
        'elev_range': None,
    }

    rgb_ext = '.png'
    gray_ext = '.png'
    depth_ext = '.png'

    # Split type should be 'pbr' for images rendered with BlenderProc
    if split_type == 'pbr':
        # The photorealistic synthetic images are provided in the JPG format.
        rgb_ext = '.jpg'

    p['im_modalities'] = ['rgb', 'depth']

    # Custom dataset.
    p['scene_ids'] = list(range(1, 31))
    p['im_size'] = (720, 540)

    base_path = join(datasets_path, dataset_name)
    split_path = join(base_path, split)
    if split_type is not None:
        if split_type == 'pbr':
            # Scene_ids is changed later in the code, so this is a default value
            p['scene_ids'] = list(range(50))
        split_path += '_' + split_type

    p.update({
        # Path to the split directory.
        'split_path': split_path,

        # Path template to a file with per-image camera parameters.
        'scene_camera_tpath': join(
            split_path, '{scene_id:06d}', 'scene_camera.json'),

        # Path template to a file with GT annotations.
        'scene_gt_tpath': join(
            split_path, '{scene_id:06d}', 'scene_gt.json'),

        # Path template to a file with meta information about the GT annotations.
        'scene_gt_info_tpath': join(
            split_path, '{scene_id:06d}', 'scene_gt_info.json'),

        # Path template to a file with the coco GT annotations.
        'scene_gt_coco_tpath': join(
            split_path, '{scene_id:06d}', 'scene_gt_coco.json'),

        # Path template to a gray image.
        'gray_tpath': join(
            split_path, '{scene_id:06d}', 'gray', '{im_id:06d}' + gray_ext),

        # Path template to an RGB image.
        'rgb_tpath': join(
            split_path, '{scene_id:06d}', 'rgb', '{im_id:06d}' + rgb_ext),

        # Path template to a depth image.
        'depth_tpath': join(
            split_path, '{scene_id:06d}', 'depth', '{im_id:06d}' + depth_ext),

        # Path template to a mask of the full object silhouette.
        'mask_tpath': join(
            split_path, '{scene_id:06d}', 'mask', '{im_id:06d}_{gt_id:06d}.png'),

        # Path template to a mask of the visible part of an object silhouette.
        'mask_visib_tpath': join(
            split_path, '{scene_id:06d}', 'mask_visib',
            '{im_id:06d}_{gt_id:06d}.png'),
    })

    return p


def get_present_scene_ids(dp_split):
    """Returns ID's of scenes present in the specified dataset split.

    :param dp_split: Path to a folder with datasets.
    :return: List with scene ID's.
    """
    scene_dirs = [d for d in glob.glob(os.path.join(dp_split['split_path'], '*'))
                  if os.path.isdir(d)]
    scene_ids = [int(os.path.basename(scene_dir)) for scene_dir in scene_dirs]
    scene_ids = sorted(scene_ids)
    return scene_ids
