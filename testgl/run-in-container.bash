
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

echo "DRI in container:"
ls -alr /dev/dri
echo "DISPLAY in container: '${DISPLAY}'"

ls -al /tmp/.X11-unix

. ${pyenv-/pyenv/venv/bin/activate}
cd testgl

python cube.py
python hello_pybullet.py
