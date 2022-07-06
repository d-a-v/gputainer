
Either
- Clone submodules: `git submodule update --init --recursive`

  `diff` with original repo [focused on environment.yaml](https://github.com/ylabbe/cosypose/compare/master...d-a-v:cosypose:torch18#diff-0ff3348ae0de662ef053d7de057a7a92c091a09fc3d245c8da02bec028ca3d0b)

or
- Get your own version of cosyppose and adapt `cosypose/environment.yaml` for torch-1.8, or use [this version](https://github.com/d-a-v/cosypose/blob/torch18/environment.yaml)

Then

- Get your own `cosypose/local_data/`
- Run `./10-cosy.build-nvidia-515` which takes a while
- Run `./20-runvnc`

  Python dependences will be compiled on the first run.

  For next runs, they will be reused. They can be rebuilt by removing them first with `./02-rm-cosypose-python-builds`

- Check the `RUN IT NOW` command in the console
