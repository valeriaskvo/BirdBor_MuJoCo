import mujoco
from mujoco import viewer
import numpy as np
import matplotlib.pyplot as plt

xml_file = 'xml_examples/actuator_example.xml'
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)
mujoco.mj_resetData(model, data)

x = []
dx = []
t = []

with viewer.launch_passive(model, data) as viewer_pass:
  while viewer_pass.is_running():
    for i in range(1000):
        data.qpos = [0, np.sin(data.time * 0.01)]
        mujoco.mj_step(model, data)
        x.append(data.qpos.copy())
        dx.append(data.qvel.copy())
        t.append(data.time)
    viewer_pass.sync()
    
# plt.plot(t, x)
# plt.show()

# plt.plot(t, dx)
# plt.show()
# print(data.qvel)



viewer.launch(model, data)