# Data generation

This repository enables you to generate synthetic images of objects specified as CAD models and then use them to train CosyPose.

It is adapted to .ply CAD models and the scripts are made specifically for clusters running with Slurm.

## Preamble
---
### Data
--- 

Download the [bop datasets](https://bop.felk.cvut.cz/datasets/). Then, point to the directory storing these data, as indicated in the next section. These dataset can be used to train CosyPose but are also used as decoy during the data generation.


### Configuration
---
The first step is to configure the `environment.sh` file according to your environment. In particular:

```
data_folder=/path/to/the/data_folder/
```

Once these variables are configured, they are used in different files, in particular `.config` file. Please ensure that you follow the indicated structures or adapt this file.

In particular, it important to notice that the `bop_parent_path` points to the bop dataset, used as decoy in this context, and to the `custom_cad_models_path`, which points to the target custom data to train on.


## Execution
---


Then, as the other modules of this repository, the execution follows these steps:

1. Build
```
./1-build-nvidia-515
```
2. Run the data generation process
```
./2-initial-submitter
```
3. Run the post-generation process
```
sbatch 20-slurm.sbatch
```
An utilitary cleaning script is provided, to delete temporary files.
```
./00-clean
```

## Explanation
--- 

### 1. Build

During the `2.Build` phase, the container is build following the same reasoning as the others modules. The main difference happens in the `user-postinstall`. 

First, environment variables are set in the container.
Then, blender is installed. Feel free to modify the version of Blender according to the releases. However, we cannot garanty that it will not affect other parts of the system.

Finally, [blenderproc](https://github.com/DLR-RM/BlenderProc) is configured and used to download necessary textures.

### 2. Data generation process

This process is run in parallel to speed up the computation. Different slurm jobs will be run.

### 3. Post-generation process


The data generated in each of these batches are gathered in a single folder, to make them usable for the training phase.
