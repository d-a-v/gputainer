import blenderproc as bproc
import json
from pathlib import Path
from glob import glob
import argparse
import os
import numpy as np

from cad2cosypose import read_config_file

# Import env variables
cfg = read_config_file()

parser = argparse.ArgumentParser()
parser.add_argument('--bop-parent-path', dest='bop_parent_path', type=str,
                    default=cfg["bop_parent_path"],
                    help="Path to the bop datasets parent directory")
parser.add_argument('--custom-dataset-path', dest='custom_dataset_path', type=str,
                    default=cfg["custom_cad_models_path"],
                    help="Path to the bop datasets parent directory")
parser.add_argument('--cc-textures-path', dest='cc_textures_path', type=str,
                    default=cfg["cc_textures_dir"],
                    help="Path to downloaded cc textures")
parser.add_argument('--output-dir', dest='output_dir', type=str,
                    default=cfg["output_dir"],
                    help="Path to where the final files will be saved ")
parser.add_argument('--num-scenes', dest='num_scenes', type=int,
                    default=cfg["number_scenes"],
                    help="How many scenes with 25 images each to generate")
args = parser.parse_args()

bproc.init()

############################################
# CUSTOM SETTINGS FOR CUSTOM DATASET
CUSTOM_SHADERS = {
    "Specular": np.random.uniform(0.2, 0.5),
    "Metallic": np.random.uniform(0., 0.15),
    "Roughness": np.random.uniform(0.025, 0.2),
    "Transmission": np.random.uniform(0.1, 0.4),
}
SCALE_CUSTOM_OBJS = [1e-3, 1e-3, 1e-3]
NB_TARGET_OBJS_PER_SCENE = 20

BOP_DISTRACTORS = {
    'tless': 4,
    'itodd': 2,
    'ycbv': 2,
    'hb': 2,
}
############################################

DATASET_NAME = cfg["dataset_name"]


def get_custom_category_ids(dir: str):
    CATEGORIES = glob(dir + "/**/**.ply")
    CATEGORIES.sort()

    CATEGORY_IDS = {category: id for id, category in enumerate(CATEGORIES)}

    output_dir = Path(args.output_dir) / DATASET_NAME
    os.makedirs(output_dir, exist_ok=True)
    with open(output_dir / "categories.json", "w") as f:
        json.dump(CATEGORY_IDS, f)

    return CATEGORY_IDS


def load_obj_from_path(path: str, category_id, scale=1.0):
    obj = bproc.loader.load_obj(path, cached_objects=[])[0]
    obj.set_cp("category_id", category_id)
    obj.set_scale(scale)
    obj.set_cp("bop_dataset_name", DATASET_NAME)
    return obj


def load_objs_from_dir(dir: str, scale=1.0):
    category_ids = get_custom_category_ids(dir)
    return [
        load_obj_from_path(path, category_id, scale)
        for path, category_id in category_ids.items()
    ]


# load custom objects into the scene
target_objs = load_objs_from_dir(
    args.custom_dataset_path, scale=SCALE_CUSTOM_OBJS)


# load distractor bop objects
dist_bop_objs = {
    dataset: bproc.loader.load_bop_objs(
        bop_dataset_path=os.path.join(args.bop_parent_path, dataset), mm2m=True)
    for dataset in BOP_DISTRACTORS.keys()
}

# load BOP datset intrinsics
bproc.loader.load_bop_intrinsics(
    bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'))

# set shading and hide objects
for obj in (target_objs + [obj for _, objs in dist_bop_objs.items() for obj in objs]):
    obj.set_shading_mode('auto')
    obj.hide(True)

# create room
room_planes = [bproc.object.create_primitive('PLANE', scale=[2, 2, 1]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[
                                             0, -2, 2], rotation=[-1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[
                                             0, 2, 2], rotation=[1.570796, 0, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[
                                             2, 0, 2], rotation=[0, -1.570796, 0]),
               bproc.object.create_primitive('PLANE', scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0])]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape='BOX', mass=1.0,
                           friction=100.0, linear_damping=0.99, angular_damping=0.99)

# sample light color and strenght from ceiling
light_plane = bproc.object.create_primitive(
    'PLANE', scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name('light_plane')
light_plane_material = bproc.material.create('light_material')

# sample point light on shell
light_point = bproc.types.Light()
light_point.set_energy(100)

# load cc_textures
cc_textures = bproc.loader.load_ccmaterials(args.cc_textures_path)

# Define a function that samples 6-DoF poses


def sample_pose_func(obj: bproc.types.MeshObject):
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.6])
    obj.set_location(np.random.uniform(min, max))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

for i in range(args.num_scenes):
    print(target_objs)
    # Sample bop objects for a scene
    sampled_target_objs = list(np.random.choice(
        target_objs, size=NB_TARGET_OBJS_PER_SCENE))
    sampled_distractor_bop_objs = list()
    for dataset, nb in BOP_DISTRACTORS.items():
        sampled_distractor_bop_objs += list(np.random.choice(
            dist_bop_objs[dataset], size=nb, replace=False))

    # Randomize materials and set physics
    for obj in (sampled_target_objs + sampled_distractor_bop_objs):
        mat = obj.get_materials()[0]
        if obj.get_cp("bop_dataset_name") in ['itodd', 'tless']:
            grey_col = np.random.uniform(0.1, 0.9)
            mat.set_principled_shader_value(
                "Base Color", [grey_col, grey_col, grey_col, 1])
        mat.set_principled_shader_value("Roughness", np.random.uniform(0, 0.5))
        if obj.get_cp("bop_dataset_name") == 'itodd':
            mat.set_principled_shader_value(
                "Metallic", np.random.uniform(0.5, 1.0))
        if obj.get_cp("bop_dataset_name") == 'tless':
            mat.set_principled_shader_value(
                "Specular", np.random.uniform(0.3, 1.0))
            mat.set_principled_shader_value(
                "Metallic", np.random.uniform(0, 0.5))
        if obj.get_cp("bop_dataset_name") == DATASET_NAME:
            for name, value in CUSTOM_SHADERS.items():
                mat.set_principled_shader_value(name, value)
        obj.enable_rigidbody(True, mass=1.0, friction=100.0,
                             linear_damping=0.99, angular_damping=0.99)
        obj.hide(False)

    # Sample two light sources
    light_plane_material.make_emissive(emission_strength=np.random.uniform(3, 6),
                                       emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]))
    light_plane.replace_materials(light_plane_material)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                                   elevation_min=5, elevation_max=89)
    light_point.set_location(location)

    # sample CC Texture and assign to room planes
    random_cc_texture = np.random.choice(cc_textures)
    for plane in room_planes:
        plane.replace_materials(random_cc_texture)

    # Sample object poses and check collisions
    bproc.object.sample_poses(objects_to_sample=sampled_target_objs + sampled_distractor_bop_objs,
                              sample_pose_func=sample_pose_func,
                              max_tries=1000)

    # Physics Positioning
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3,
                                                      max_simulation_time=10,
                                                      check_object_interval=1,
                                                      substeps_per_frame=20,
                                                      solver_iters=25)

    # BVH tree used for camera obstacle checks
    bop_bvh_tree = bproc.object.create_bvh_tree_multi_objects(
        sampled_target_objs + sampled_distractor_bop_objs)

    cam_poses = 0
    while cam_poses < int(cfg["number_camera_poses_per_scene"]):
        # Sample location
        location = bproc.sampler.shell(center=[0, 0, 0],
                                       radius_min=0.65,
                                       radius_max=0.94,
                                       elevation_min=5,
                                       elevation_max=89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(
            sampled_target_objs, size=15, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(
            location, rotation_matrix)

        # Check that obstacles are at least 0.3 meter away from the camera and make sure the view interesting enough
        if bproc.camera.perform_obstacle_in_view_check(cam2world_matrix, {"min": 0.3}, bop_bvh_tree):
            # Persist camera pose
            bproc.camera.add_camera_pose(cam2world_matrix, frame=cam_poses)
            cam_poses += 1

    # render the whole pipeline
    data = bproc.renderer.render()

    # Write data in bop format
    bproc.writer.write_bop(os.path.join(args.output_dir),
                           target_objects=sampled_target_objs,
                           dataset=DATASET_NAME,
                           depth_scale=0.1,
                           depths=data["depth"],
                           colors=data["colors"],
                           color_file_format="JPEG",
                           ignore_dist_thres=10)

    for obj in (sampled_target_objs + sampled_distractor_bop_objs):
        try:
            obj.disable_rigidbody()
            obj.hide(True)
        except Exception as e:
            print(e)
