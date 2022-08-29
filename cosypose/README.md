
## Notes

`apptainer` must be installed on the gpu server.

`turbovnc` client must be installed on the viewer host.

The generated image is an environment to be able to run `cosypose`.
`cosypose` will not be included in the generated image, it stays external and modifiable.

Once generated, directories look like:
```
...gputainer/cosypose$ ls -l generated/
drwxr-xr-x  4 x x 4096 Aug  5 15:37 PYTHAINER
drwxr-xr-x 25 x x 4096 Aug  5 14:17 cosypose.sifdir
```

See below for running it.

If desired, once transformed into a plain SIF file, directories look like:
```
...gputainer/cosypose>ls -lh transportable/
total 8818396
lrwxrwxrwx 1 x x   11 Aug  5 15:32 cosypose -> ../cosypose
-rwxr-xr-x 1 x x 4.8K Aug  5 15:31 cosypose.runme
-rw-r--r-- 1 x x 8.5G Aug  5 15:37 cosypose.sif
-rwxr-xr-x 1 x x 1.9K Aug  5 15:31 run-in-container
```

`cosypose` directory (or in this example a symbolic link) is your cosypose git repository and must be in the local directory.

## Operations
 
- First, execute `./00-setup-env`
- Get your own cosypose's "`local_data/`" and set its path into `00-setup-env.vars.sh`
- Run `./10-cosy.build-nvidia-515` which takes a while
- Run `./20-runvnc`

  Python dependencies inside `cosypose` will be compiled on the first run.

  For next runs, they will be reused. They can be rebuilt by removing them first with `./01-rm-cosypose-python-builds`
