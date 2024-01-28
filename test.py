import mujoco
from mujoco import viewer
from numpy import sin, arcsin
import matplotlib.pyplot as plt
import time

xml_file = 'BirdBot.xml'
model = mujoco.MjModel.from_xml_path(xml_file)
data = mujoco.MjData(model)
x = []
dx = []
t = []

SIMULATION_TIME = 10

shift = 0.8
w = 5
phase = arcsin(shift)
tc_0 = 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    viewer.sync()
    while viewer.is_running() and time.time() - start < SIMULATION_TIME:
        step_start = time.time()

        # if step_start - start > 1:
        #     if tc_0 == 0:
        #         tc_0 = step_start
        #     data.qpos[2] = shift - sin((step_start - tc_0)*w + phase)

        data.qpos[0] = 0
        data.qpos[1] = 0.05*sin((step_start - tc_0)*w + phase)
        mujoco.mj_step(model, data)

        with viewer.lock():
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(data.time % 2)

        viewer.sync()
        current_time = time.time()
        time_until_next_step = model.opt.timestep - (current_time - step_start)
        x.append(data.qpos.copy())
        dx.append(data.qvel.copy())
        t.append(current_time - start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
viewer.close()
      
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
ax1.plot(t, x)
ax1.set_ylabel('Position')
ax1.grid()
ax2.plot(t, dx)    
ax2.set_ylabel('Velocity')
ax2.grid()
ax2.set_xlabel('Time')
plt.show()