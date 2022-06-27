
set -ex

# $HOME can be an artificial place, and
# - it is not kept up through apptainer
# - it is forbidden to set it up via APPTAINERENV_HOME
# so it is trnasmitted through APPTAINERENV_MYHOME:
if [ -z "${MYHOME}" ]; then
    echo "\$MYHOME (from APPTAINERENV_MYHOME) should be set!"
else
    export HOME=${MYHOME}
fi

. /pyenv/conda/etc/profile.d/conda.sh
conda activate /pyenv/app

# PATH and LD_LIBRARY_PATH from python_path ?
export PATH=/pyenv/app/bin:${PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/pyenv/app/lib/python3.7/site-packages/torch/lib

echo inside apptainer DISPLAY=${DISPLAY}
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo XAUTHORITY=${XAUTHORITY}
echo PWD=$(pwd)
ls -alr /dev/dri

python -c 'import torch;print("torch.__version__=", torch.__version__, "\ntorch.cuda.device_count()=", torch.cuda.device_count(), "\ntorch.cuda.is_available()=", torch.cuda.is_available(), "\ntorch.version.cuda=", torch.version.cuda, "\ntorch.backends.cudnn.version()=", torch.backends.cudnn.version());'
#vglrun -d /dev/dri/card1 python -c 'import torch;print("torch.__version__=", torch.__version__, "\ntorch.cuda.device_count()=", torch.cuda.device_count(), "\ntorch.cuda.is_available()=", torch.cuda.is_available(), "\ntorch.version.cuda=", torch.version.cuda, "\ntorch.backends.cudnn.version()=", torch.backends.cudnn.version());'

export CUDA_LAUNCH_BLOCKING=1
cd cosypose

# build but not setting up because apptainer's sif file might be not writable
# then setting PATH below
[ -d build ] || python setup.py build
(cd deps/bullet3; [ -d build ] || python setup.py build)
(cd deps/job-runner; [ -d build ] || python setup.py build)
export PYTHONPATH=$(pwd)/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/bullet3/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/job-runner/build/lib.linux-x86_64-cpython-37:${PYTHONPATH}
export LD_LIBRARY_PATH=$(pwd)/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/bullet3/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/job-runner/build/lib.linux-x86_64-cpython-37:${LD_LIBRARY_PATH}

xeyes & eyes=$!
#python -m cosypose.scripts.run_cosypose_eval --config tless-siso --debug
python -m cosypose.scripts.prediction_script
kill $eyes
