
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

- Configure Apptainer users by updating `/etc/subuid` and `/etc/subgid`

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
