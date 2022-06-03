
. /conda/conda/etc/profile.d/conda.sh
conda activate /conda/app
export PATH=/conda/app/bin:${PATH}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/conda/app/lib/python3.7/site-packages/torch/lib
env|grep -i PYTHON

# PATH and LD_LIBRARY_PATH from python_path ?
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}

# vglrun seems to need BOTH '-d :1' and 'DISPLAY=:1' ?
vglrun -d ${DISPLAY} python -c 'import torch;print("torch.cuda.device_count()=", torch.cuda.device_count(), "\ntorch.cuda.is_available()=", torch.cuda.is_available(), "\ntorch.version.cuda=", torch.version.cuda, "\ntorch.backends.cudnn.version()=", torch.backends.cudnn.version());'

export CUDA_LAUNCH_BLOCKING=1
cd cosypose

export PYTHONPATH=$(pwd)/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/bullet3/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/job-runner/build/lib.linux-x86_64-cpython-37:${PYTHONPATH}
export LD_LIBRARY_PATH=$(pwd)/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/bullet3/build/lib.linux-x86_64-cpython-37:$(pwd)/deps/job-runner/build/lib.linux-x86_64-cpython-37:${LD_LIBRARY_PATH}

[ -d build ] || python setup.py build
(cd deps/bullet3; [ -d build ] || python setup.py build)
(cd deps/job-runner; [ -d build ] || python setup.py build)

vglrun -d ${DISPLAY} python -m cosypose.scripts.run_cosypose_eval --config tless-siso --debug
#??vglrun -d ${DISPLAY} python -m cosypose.scripts.prediction_script
#???vglrun -d ${DISPLAY} python -m cosypose.scripts.switches_powerstrips1
