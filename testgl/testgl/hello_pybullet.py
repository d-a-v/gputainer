import pybullet as p
import time
import pybullet_data


physicsClient = p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")
cubeStartPos = [0, 0, 1]
cubeStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
boxId = p.loadURDF("r2d2.urdf", cubeStartPos, cubeStartOrientation)
cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)


start_ns = time.time_ns()
duration_ns = 5*1000*1000*1000;
while time.time_ns() - start_ns < duration_ns:
    time.sleep(0.05)  # Time in seconds.
    p.stepSimulation()
