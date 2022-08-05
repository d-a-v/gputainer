
# Headless & batch runtime environment helper for building and starting GPU applications requiring a display, using Apptainer (ex-singularity)

This environment is designed to help running GPU & EGL applications in a SLURM environment.

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

- [This is no longer necessary](https://github.com/apptainer/apptainer/releases/tag/v1.1.0-rc.1) Configure Apptainer users by updating `/etc/subuid` and `/etc/subgid`

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
directories `PYTHAINER/` and `<app>.sif/`.  `<app>.sif` should be a file in
a real world, and can later be converted, but is really a directory.  This
directory is rebuilded everytime when restarting the `build` script, and is
the root filesystem of the container.

`PYTHAINER` is not rebuilded everytime, it has to be removed for that.  This
allows to incrementally tune the build scripts.  This directory is the
python root container for `conda` or `PyPI` and can sometimes take ages to
be fully rebuilt.

When running a container with `./xxxx-runvnc`, the `run-in-container` script is started.

To manually check the container state after a build and run an `xterm`
instead, here is an example with the `testgl` example:

- On the gpu server: `./2-runvnc bash`

  Wait for the `RUN IT` message and copy the `vncviewer` line

- On your local host, paste the command

- In the opened xterm inside vnc,
  - start another `xterm -geometry 160x60 bash`
  - or try running `icewm` which is also installed, and start a console

- In the new console, type `cat run-in-container` and copy-paste the few lines which activate the python environment (if/generally applicable).

- Play, noting that:
    - you are not root in the container filesystem (no `apt install` allowed)
    - you can modify the python environment (`pip install`) and this will be persistent
      but not reproducible until you add the commands in the `user-postinstall` script.

## Convert container to file

A script called `../__gputainer/sifdir-to-sif` will build and store
`transportable/<app>.sif` (and `<app>.runme` along with user's
`run-in-container`) out of `generated/` directory content.  External
directories such as repositories cloned in setup scripts are not included
in the resulting `sif` file.
