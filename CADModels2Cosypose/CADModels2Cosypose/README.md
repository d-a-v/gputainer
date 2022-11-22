# From CAD models to photorealistic images to CosyPose

This repository enables you to generate synthetic images of objects specified as CAD models and then train [CosyPose](https://github.com/ylabbe/cosypose) on it in a minimum of command lines.

It is adapted to .ply CAD models and the scripts are made specifically for clusters running with Slurm.

_Legend:_ :globe_with_meridians: = requires an Internet connection, :hourglass_flowing_sand: = may take hours.

## Data generation with [BlenderProc](https://github.com/DLR-RM/BlenderProc)

1. :globe_with_meridians: Clone this repository recursively:

```
git clone --recurse-submodules https://github.com/thomaschabal/CADModels2Cosypose.git
```

2. :globe_with_meridians: Install [Blender](https://www.blender.org/download/) if you don't already have it. You may use `make download_blender30` for that, after defining the variable `blender_parent_path` in `.config`.
3. Define your env variables in `.config`.
4. Install Python packages and download textures for Blender:

```
make setup
```

5. Optionally modify the script `generate_data.py` to fit your setup (in terms of lighting, object sizes, etc). Only change the first lines of the file for that.
6. :hourglass_flowing_sand: Generate the batches of images (using Slurm to submit jobs):

```
make generation
```

7. Once all the images are generated, gather them in a single folder and complete the dataset:

```
make post_generation
```

(Optionally add symmetries to the generated `models_info.json`)

#### Tips and possible issues

<details>

<summary>Click here for details...</summary>

- When installing Blender, check that _blender-folder_/_version-number_/python/include/_python-number_/ includes a `Python.h` file, otherwise you may get errors of missing `Python.h` when installing some pip packages. If the file does not exist, follow the procedure described in the [second option of this answer](https://blender.stackexchange.com/a/107381).
- There can be problems with the download of textures during `make setup`. In that case, you may change the cc_textures_dir value in .config, launch `make download`, and then manually merge the downloaded textures.

</details>

## Training [CosyPose](https://github.com/ylabbe/cosypose)

1. :globe_with_meridians: Install CosyPose and its submodules: `git submodules update --init` (TO COMPLETE). Beware, installing the package `./deps/bullet3` requires to be on a GPU, otherwise it will fail to build the wheel.

```
make install_cosypose
```

2. Activate the conda environment for CosyPose:

```
conda activate cosypose
```

3. Adapt the `job-runner-config.yaml` file with your own settings, then run `runjob-config job-runner-config.yaml`.
4. :hourglass_flowing_sand: Train the detector and pose estimators, coarse and refine networks, in parallel:

```
runjob --ngpus=32 python -m cad2cosypose.run_detector_training --config custom
runjob --ngpus=32 python -m cad2cosypose.run_pose_training --config custom-coarse
runjob --ngpus=32 python -m cad2cosypose.run_pose_training --config custom-refiner
```

#### Tips and possible issues

<details>

<summary>Click here for details...</summary>

- If your CAD models are not in .ply format, [Meshlab](https://www.meshlab.net) may help you to convert them.
- You may check your dataset was well recorded by running CosyPose's [render_dataset.ipynb notebook](https://github.com/ylabbe/cosypose/blob/master/notebooks/render_dataset.ipynb).

</details>

# TODO for open-sourcing

- Change files to take .obj files as input instead of .ply (blenderProc, then bop_toolkit for dataset_cfgs and generation of data)
- Test the pipeline on few images from end to end
- Add arguments to give to functions in this README
- Add a link to models_info.json to get an example of how adding symmetries in the readme
- Add architecture of data and models
- Add pictures of how results look like
- Add requirements (objects in .ply or .obj)
- Factorize generate_data/generate_data_assembly and put these files in a generation folder
- Change name of dataset
