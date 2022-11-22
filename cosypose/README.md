
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

See 'Operations' below for running it.

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

## Operations

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
