## Notes

See [main readme](../README.md) for prerequisites.

Walkthrough is described below (see 'Operations').

### Testing container

The generated image is an environment to be able to run `cosypose`.
`cosypose` will not be included in the generated image, it stays external and modifiable.

The `run-in-container` script is also external from the apptainer image: it can be modified before running the container.
By default it runs the example "Reproducing single-view results - YCB-Video".

Once generated, directories look like:
```
...gputainer/cosypose$ ls -l generated/
drwxr-xr-x  4 x x 4096 Aug  5 15:37 PYTHAINER
drwxr-xr-x 25 x x 4096 Aug  5 14:17 cosypose.sifdir
```

The repository has two main usages : `Inference` and `Training`

# Inference


## Using a Prebuild container

### Transportable container

If desired, the container can be transformed into a plain single SIF file.

Still, directory `cosypose/` and script `./run-in-container` are being kept
out from the container to allow further modification without rebuilding
everything.

`transportable/` will look like:
```
...gputainer/cosypose$ ls -lh transportable/
lrwxrwxrwx 1 x x   11 Nov 22 16:11 cosypose -> ../cosypose
-rwxr-xr-x 1 x x 6.8K Nov 22 16:11 cosypose.runme
-rw-r--r-- 1 x x 8.2G Nov 22 16:16 cosypose.sif
-rwxr-xr-x 1 x x 2.2K Nov 22 16:11 run-in-container
-rw-r--r-- 1 x x  196 Nov 22 16:25 slurm.sbatch
```

Thus, the procedure is the following:

1- Clone cosypose repository inside transportable
2- `cd cosypose` and `git submodule update --init --recursive`
3- `for p in ../../patches/*.patch; do patch -Np1 < ${p}; done`
4- copy `20-slurm.sbatch` and modify it `./20-runvnc "$@"` --> `./cosypose.runme "$@"`

## Operations : building and testing

### Testing container

#### Python dependencies

- First, execute `./00-setup-env`
- Get your own cosypose's "`local_data/`" and set its path into `00-setup-env.vars.sh`
- Run `./10-cosy.build-nvidia-515` which takes a while
- Run `./20-runvnc`

#### Python dependencies for `cosypose/`

Python dependencies inside `cosypose/` will be compiled on the first run.

For next runs, they will be reused.

They can be rebuilt by removing them first with `./01-rm-cosypose-python-builds`

### Transportable container

It is possible to create a Transportable container, once the container has been build.

The transportable SIF file is made from the `generated/` directory so the testing container must be built first.

The `transportable/` directory is created by calling `../__gputainer/sifdir-to-sif`.

*Important notice*
==================

After the first run, the `cosypose/` directory is prepared.

If the transportable image is moved elesewhere, cloning `cosypose/` directory will not be enough to run it.
It has to be:
- cloned
- with submodules updated
- patched (see `patches/`)
- with `local_data/` configured

# Training

To train the model, one must first generate the data. To do so, please refer to [CADModels2Cosypose](https://github.com/d-a-v/gputainer/tree/master/CADModels2Cosypose). Once you have generated the data, configure `custom_training/custom_training/cosypose_config.py`. In particular :

- `GENERATED_DS_DIR` is the name of the dataset which should correspond to the folder's name of the generated dataset (e.g. `tless`)
- `CUSTOM_DS_DIR` is the path to the training data (e.g `path/to/tless`)
- `CAD_DATASET_NAME` and `CAD_MODELS_DIR` follows the same principle, with the urdf dataset

Once this file is configured, you can run the following command:

`Warning: you must have build the container before running this command` 

```
./2-initial-submitter
``` 



