
# Runtime headless & batch environment helper using Apptainer for GPU applications requiring a display

These helpers are desinged to help running GPU & EGL applications in a SLURM environment.

## Prerequisites on a debian-like environment

- Get and install

    - [apptainer](https://github.com/apptainer/apptainer/releases/download/v1.0.2/apptainer_1.0.2_amd64.deb) (or [latest release](https://github.com/apptainer/apptainer/releases))
    - [VirtualGL](https://www.virtualgl.org) (debian package in [this repository](__gputainer/debs))
    - [TurboVNC](https://www.turbovnc.org) (debian package in [this repository](__gputainer/debs))

- Configure VirtualGL with [/opt/VirtualGL/bin/vglserver_config](https://rawcdn.githack.com/VirtualGL/virtualgl/3.0.1/doc/index.html#hd006)

    The only tested setup so far was to enable the three insecure questions:

        - select option 1
        - answer yes
        - answer yes
        - answer yes

    Then restart gdm / lightdm.

- Configure users for Apptainer by updating `/etc/subuid` and `/etc/subgid`:
  `(cd __gputainer; ./apptainer-temp-add-etc-subid user)`

## Check the installation

    - `ssh the-gpu-server`
    - `cd /path/to/gputainer/testgl`
    - `./1-build-nvidia-510`
    - `./2-runvnc`
      
        A short while after runningthe start script, a message will display :
      
        "please run" `/opt/TurboVNC/bin/vncviewer -password 11111 -noreconnect -nonewconn -scale auto the-gpu-server:n`
      
        This command must be copy-pasted on your console to see the application running on the GPU
