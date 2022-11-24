import blenderproc as bproc
import json
from pathlib import Path
from glob import glob
import argparse
import os
import numpy as np
from shapely.geometry import Polygon

parser = argparse.ArgumentParser()
parser.add_argument('--bop-parent-path', dest='bop_parent_path', type=str,
                    help="Path to the bop datasets parent directory")
parser.add_argument('--custom-dataset-path', dest='custom_dataset_path', type=str,
                    help="Path to the bop datasets parent directory")
parser.add_argument('--cc-textures-path', dest='cc_textures_path', type=str, default="resources/cctextures",
                    help="Path to downloaded cc textures")
parser.add_argument('--output-dir', dest='output_dir', type=str,
                    help="Path to where the final files will be saved ")
parser.add_argument('--num-scenes', dest='num_scenes', type=int, default=2000,
                    help="How many scenes with 25 images each to generate")
parser.add_argument('--stack', action='store_true',
                    help="Stack objects instead of randomizing their initial poses in the air")
args = parser.parse_args()

bproc.init()

############################################
# CUSTOM SETTINGS FOR CUSTOM DATASET
CUSTOM_SHADERS = {
    "Specular": np.random.uniform(0.2, 0.5),
    # "Metallic": np.random.uniform(0., 0.08),
    "Metallic": np.random.uniform(0., 0.15),
    # "Roughness": np.random.uniform(0.075, 0.125),
    "Roughness": np.random.uniform(0.025, 0.2),
    # "Transmission": np.random.uniform(0.2, 0.3),
    "Transmission": np.random.uniform(0.1, 0.4),
}
SCALE_CUSTOM_OBJS = [1e-3, 1e-3, 1e-3]
NB_CAMERAS_POSES = 25
# NB_CAMERAS_POSES = 3
DATASET_NAME = "custom"
STACK_PROBA = 0.7
NB_TARGET_OBJS_PER_SCENE = 20
MIN_NB_OBJS_ON_TABLE = 4
NOISE_ROT_PROPORTION = 0.3
NOISE_TRANS_PROPORTION = 0.5
############################################


def get_custom_category_ids(dir: str):
    CATEGORIES = glob(dir + "/**/**.obj")
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
# tless_dist_bop_objs = bproc.loader.load_bop_objs(
#     bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'), mm2m=True)
# itodd_dist_bop_objs = bproc.loader.load_bop_objs(
#     bop_dataset_path=os.path.join(args.bop_parent_path, 'itodd'), mm2m=True)
# ycbv_dist_bop_objs = bproc.loader.load_bop_objs(
#     bop_dataset_path=os.path.join(args.bop_parent_path, 'ycbv'), mm2m=True)
# hb_dist_bop_objs = bproc.loader.load_bop_objs(
#     bop_dataset_path=os.path.join(args.bop_parent_path, 'hb'), mm2m=True)

# load BOP datset intrinsics
# TODO: Check camera intrinsics
bproc.loader.load_bop_intrinsics(
    bop_dataset_path=os.path.join(args.bop_parent_path, 'tless'))

# set shading and hide objects
# for obj in (target_objs + tless_dist_bop_objs + itodd_dist_bop_objs + ycbv_dist_bop_objs + hb_dist_bop_objs):
for obj in (target_objs):
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


def get_pose_uniform():
    min = np.random.uniform([-0.3, -0.3, 0.0], [-0.2, -0.2, 0.0])
    max = np.random.uniform([0.2, 0.2, 0.4], [0.3, 0.3, 0.12])
    xyz = np.random.uniform(min, max)
    rot_euler = bproc.sampler.uniformSO3()
    return xyz, rot_euler


def sample_pose_uniform_func():
    xyz, rot_euler = get_pose_uniform(obj)
    obj.set_location(xyz)
    obj.set_rotation_euler(rot_euler)


def is_collision(bbox_1, bbox_2):
    def make_polygon(bbox):
        max_z = np.max(bbox[:, 2])
        lines = np.array([bbox[id, :2]
                          for id in range(bbox.shape[0])
                          if bbox[id, 2] > max_z - 1e-5])

        right_idx = np.argmax([lines[:, 0]])
        left_idx = np.argmin([lines[:, 0]])
        up_idx = np.argmax([lines[:, 1]])
        down_idx = np.argmin([lines[:, 1]])
        return Polygon([lines[right_idx], lines[up_idx], lines[left_idx], lines[down_idx]])

    polygon_1 = make_polygon(bbox_1)
    polygon_2 = make_polygon(bbox_2)
    intersection = polygon_1.intersection(polygon_2)
    return intersection.area > 0


def get_half_height_obj(obj):
    bbox = obj.get_bound_box()
    zs = bbox[:, 2]
    height = np.max(zs) - np.min(zs)
    return height / 2


def is_any_collision(obj, id):
    return any([is_collision(obj.get_bound_box(), obj_else.get_bound_box()) for obj_else in sampled_target_objs[:id]])


def get_stacked_target_poses(sampled_target_objs):
    def sample_rotations(random_rot, rot_under=None, noise=False):
        # Rotate like object below
        rot_base = 0.
        if rot_under is not None:
            rot_base = rot_under[2]

        # Rotate
        if random_rot:
            rot_z = np.pi * (2 * np.random.random() - 1)
        else:
            rot_z = np.pi / 2 * np.random.randint(0, 4)

        # Add noise for rotation
        amp = 1 / 16
        noise = amp * (2 * np.random.random() - 1) * np.pi if noise else 0.

        return (
            np.pi / 2 * np.random.randint(0, 4),
            np.pi / 2 * np.random.randint(0, 4),
            rot_base + rot_z + noise)

    def sample_xyz(obj, xyz_under=None, obj_under=None, noise=False):
        # Place from object under
        if xyz_under is None:
            min = np.random.uniform([-0.3, -0.3], [-0.2, -0.2])
            max = np.random.uniform([0.2, 0.2], [0.3, 0.3])
            xy = np.random.uniform(min, max)
            z_under = 0.
        else:
            xy = xyz_under[:2]
            z_under = xyz_under[2] + get_half_height_obj(obj_under)

        # Add noise in xy
        if noise:
            amp = 0.005
            xy = (
                xy[0] + amp * (2 * np.random.random() - 1),
                xy[1] + amp * (2 * np.random.random() - 1),
            )

        half_height = get_half_height_obj(obj)
        return (*xy, z_under + half_height)

    CLEAR_OBJS = []
    for id, obj in enumerate(sampled_target_objs):
        if id >= MIN_NB_OBJS_ON_TABLE and np.random.random() < STACK_PROBA:
            # Stack object
            possible_ids = [idx for idx,
                            clear in enumerate(CLEAR_OBJS) if clear]
            idx_obj_under = np.random.choice(possible_ids)

            obj_under = sampled_target_objs[idx_obj_under]
            xyz_under = obj_under.get_location()
            rot_under = obj_under.get_rotation()

            obj.set_rotation_euler(sample_rotations(
                random_rot=False, rot_under=rot_under, noise=np.random.random() < NOISE_ROT_PROPORTION))
            obj.set_location(sample_xyz(obj, xyz_under, obj_under,
                             noise=np.random.random() < NOISE_TRANS_PROPORTION))

            CLEAR_OBJS[idx_obj_under] = False

        else:
            # Randomly place object on the ground
            obj.set_rotation_euler(sample_rotations(random_rot=True))
            obj.set_location(sample_xyz(obj))

            while is_any_collision(obj, id):
                obj.set_rotation_euler(sample_rotations(random_rot=True))
                obj.set_location(sample_xyz(obj))

        CLEAR_OBJS.append(True)


def sample_pose_func(obj: bproc.types.MeshObject):
    # If stack mode and with a given probability, stack objects
    if not args.stack:
        sample_pose_uniform_func(obj)


# activate depth rendering without antialiasing and set amount of samples for color rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

for i in range(args.num_scenes):
    # Sample target objects for a scene, with possibly several times the same object
    sampled_target_objs = list(np.random.choice(
        target_objs, size=NB_TARGET_OBJS_PER_SCENE))
    sampled_target_objs = [obj.duplicate() for obj in sampled_target_objs]
    if args.stack:
        get_stacked_target_poses(sampled_target_objs)

    sampled_distractor_bop_objs = []
    # sampled_distractor_bop_objs = list(np.random.choice(
    #     itodd_dist_bop_objs, size=2, replace=False))
    # sampled_distractor_bop_objs += list(np.random.choice(
    #     tless_dist_bop_objs, size=4, replace=False))
    # sampled_distractor_bop_objs += list(np.random.choice(
    #     ycbv_dist_bop_objs, size=2, replace=False))
    # sampled_distractor_bop_objs += list(np.random.choice(
    #     hb_dist_bop_objs, size=2, replace=False))

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
    while cam_poses < NB_CAMERAS_POSES:
        # Sample location
        location = bproc.sampler.shell(center=[0, 0, 0],
                                       radius_min=0.65,
                                       radius_max=0.94,
                                       elevation_min=5,
                                       elevation_max=89)
        # Determine point of interest in scene as the object closest to the mean of a subset of objects
        poi = bproc.object.compute_poi(np.random.choice(
            # sampled_target_objs, size=15, replace=False))
            sampled_target_objs, size=10, replace=False))
        # Compute rotation based on vector going from location towards poi
        rotation_matrix = bproc.camera.rotation_from_forward_vec(
            poi - location, inplane_rot=np.random.uniform(-3.14159, 3.14159))
        # poi - location, inplane_rot=0.)
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

# # 1 - Make sure objects are stacked well, as expected
# # 2 - Add rotations of objects around x or y axes and ensure the stacks still match what is expected
# # Check issues with order of rotations around axes
# # 3 - Add some noise on a proportion of images on the rotations and translations
# 4 - For some situations try to face stacks (see only one face)
# # 5 - Share between spread objects and stacked ones -> Not sure
# # 6 - Set back all parameters, with distractor objects..., REMOVE SEED
# # Also, check order of addition to env is randomized, i.e. not always the same objects appear first
