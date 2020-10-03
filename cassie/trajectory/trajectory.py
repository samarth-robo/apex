import numpy as np
import random
import math

"""
Agility 2 kHz trajectory
"""
class CassieTrajectory:
    def __init__(self, filepath):
        n = 1 + 35 + 32 + 10 + 10 + 10
        data = np.fromfile(filepath, dtype=np.double).reshape((-1, n))

        # states
        self.time = data[:, 0]
        self.qpos = data[:, 1:36]
        self.qvel = data[:, 36:68]

        # actions
        self.torque = data[:, 68:78]
        self.mpos = data[:, 78:88]
        self.mvel = data[:, 88:98]

    def state(self, t):
        tmax = self.time[-1]

        i = int((t % tmax) / tmax * len(self.time))

        return (self.qpos[i], self.qvel[i])

    def action(self, t):
        tmax = self.time[-1]
        i = int((t % tmax) / tmax * len(self.time))
        return (self.mpos[i], self.mvel[i], self.torque[i])

    def sample(self):
        i = random.randrange(len(self.time))
        return (self.time[i], self.qpos[i], self.qvel[i])

    def __len__(self):
        return len(self.time)


class CassieKeyframes(object):
    def __init__(self, filename, length, sim_freq=2000.0):
        self.qpos = np.fromfile(filename, dtype=np.double).reshape((-1, 35))
        self.length = length
        self.dt = 1.0/sim_freq

    def lerp(self, t):
        t = float(t) * float(len(self.qpos)-1) / self.length
        i0 = math.floor(t)
        i1 = math.ceil(t)
        w = i1 - t
        out = w*self.qpos[int(i0)] + (1.0-w)*self.qpos[int(i1)]
        return out

    def numerical_qvel(self, t):
        t0 = max(t-1.0, 0)
        t1 = min(t+1.0, self.length)
        p0 = self.lerp(t0)
        p1 = self.lerp(t1)
        v = (p1 - p0) / ((t1-t0) * self.dt)
        return v