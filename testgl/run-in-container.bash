
[ -z "${MYHOME}" ] || export HOME=${MYHOME}

echo "DRI in container:"
ls -alr /dev/dri
echo "DISPLAY in container: '${DISPLAY}'"

ls -al /tmp/.X11-unix

. ${pyenv-/pyenv/venv/bin/activate}
cd testgl

#glxgears
#vglrun -d ${DISPLAY} glxgears

python cube.py
#vglrun python pegl/tests/test_egl.py
#vglrun python pegl/tests/test_display_mock.py
