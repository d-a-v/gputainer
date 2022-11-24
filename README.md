
# Headless & batch runtime environment helper for building and starting GPU applications requiring a display, using Apptainer (ex-singularity)

This environment is designed to help running GPU & EGL applications in a SLURM environment. It is organized in different subparts ([cosypose](https://github.com/d-a-v/gputainer/tree/master/cosypose), [cupy](https://github.com/d-a-v/gputainer/tree/master/cupy), [ros_cosypose](https://github.com/d-a-v/gputainer/tree/master/ros_cosypose), [testgl](https://github.com/d-a-v/gputainer/tree/master/testgl)). The following section describes the general prerequisites. For each subpart, please refer to its dedicated readme.

## Prerequisites on a debian-like environment

- Get and install

  On the server:

  - [apptainer](https://github.com/apptainer/apptainer/releases/) (debian package in [this repository](__gputainer/debs))
  - [VirtualGL](https://www.virtualgl.org) (debian package in [this repository](__gputainer/debs))
  - [TurboVNC](https://www.turbovnc.org) server (debian package in [this repository](__gputainer/debs))

  On your local console:

  - TurboVNC (client, from the same package)

- Configure VirtualGL with [/opt/VirtualGL/bin/vglserver_config](https://rawcdn.githack.com/VirtualGL/virtualgl/3.0.1/doc/index.html#hd006)

  The only tested setup so far was to enable the three insecure questions:

  - if they are enabled: stop gdm / lightdm
  - run `/opt/VirtualGL/bin/vglserver_config`
  - select option 1
  - answer yes / no / no / no / X  (insecure, testing)
  - restart gdm / lightdm.

- [This is no longer necessary](https://github.com/apptainer/apptainer/releases/tag/v1.1.0)

  Configure Apptainer users by updating `/etc/subuid` and `/etc/subgid`

  A helper is provided:

  `(cd __gputainer; ./apptainer-temp-add-etc-subid <user>)`

## Check the installation

- (`ssh the-gpu-server`)
- `cd /path/to/gputainer/testgl`
- `./1-build-nvidia-510`
- `./2-runvnc`

A short while after running the start script, a message will display :

```
"please run":
/opt/TurboVNC/bin/vncviewer -password 11111 -noreconnect -nonewconn -scale auto the-gpu-server:n
```

This command must be copy-pasted on your local console to see the application running on the distant GPU.

## All-in-one run command

A "single command" started on user's desktop without GPU capabilities to start&watch everything running remotely:

First time, get the command to issue later on your local desktop:
```
$ ssh remote
$ cd /path/to/gputainer
$ ./runner-for-client

This script must be started on your desktop host with:

    scp ... && ./runner-for-client ... <application-name>

where application name is one of:

testgl
...

$ exit
```

When the application is builded on server, the `scp ... testgl` line can be copy-pasted on your local desktop.

## Tweaking inside the container

Upon build, the subdirectory `<app>/generated/` is created.  It contains two
directories `PYTHAINER/` and `<app>.sifdir/`.

`<app>.sifdir` should be a file, and can later be created (see below), but a
directory is preferred during the first steps to allow examining / ease
tuning inside the container.  This directory is rebuilded everytime when
restarting the `build` script, and is the root filesystem of the container.

`PYTHAINER/` is the python root container for `conda` or `PyPI` and can
sometimes take ages to be fully rebuilt.  It is not rebuilded everytime,
unless it is removed.  This allows to incrementally tune the build scripts.

When the container is started with `./xxxx-runvnc`, the `run-in-container`
script is started from inside.

To manually check a container after a build, here are the steps with the
`testgl` example:

- On the gpu server: `./2-runvnc icewm` (or `slurm 2-slurm.sbatch icewm`)

  Note the optional argument `icewm`, it is always installed inside in the
  container.

  Wait for the `RUN IT` on the console (or in `stdout.txt` when using SLURM)
  and copy the `vncviewer` line to a shell on your *desktop host* = not the
  remote GPU host.

- In the VNC client, use `icewm` menus to start an X terminal

- In the new console, type `cat run-in-container` and copy-paste the few
  lines which activate the python environment (if applicable).

- Check everything is working, noting that:
    - You are not root in the container filesystem (no `apt install` allowed)
    - You can modify the python environment (`pip install`) and this will be persistent
      but not reproducible upon rebuild until you add the commands in the
      `user-pre/postinstall` scripts or change your `environment.yaml` /
      `requirements.txt` descriptions.

## Converting the directory container to a transportable file

A callable script:
```
../__gputainer/sifdir-to-sif
```
allows to convert the `generated/` directory container into
a `transportable/<app>.sif` file, and create side files and scripts like
`transportable/<app>.runme` along with user's `run-in-container`.

Note that the `run-in-container` script, and also the external `<app>`
directory *are not included* in the resulting `sif` file on purpose, they
are respectively copied and symlinked.

Not including them allows to tune / update / work on the payload / running
code while not having to costingly fully rebuild the running environment.
