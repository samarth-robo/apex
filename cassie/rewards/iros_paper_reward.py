import numpy as np

def iros_paper_reward(self,state):
    qpos = state[:35]
    ref_pos = self.traj.qpos[(self.phase%self.phaselen)*self.simrate]
    ref_pos[0] += (self.phase//self.phaselen) * (self.traj.qpos[-1,0]-self.traj.qpos[0,0])
    ref_pos[1] = 0

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    joint_error       = 0
    com_error         = 0
    orientation_error = 0
    spring_error      = 0

    idx = [7,8,9,14,20, 21,22,23,28,34]
    # each joint pos
    joint_error = 30 * np.sum( weight * (qpos[idx] - ref_pos[idx]) ** 2 )

    # center of mass: x, y, z
    idx = [0, 1, 2]
    com_error = np.sum( (qpos[idx] - ref_pos[idx]) ** 2 )

    # COM orientation: qx, qy, qz
    idx = [4, 5, 6]
    orientation_error = np.sum( (qpos[idx] - ref_pos[idx]) ** 2 )

    # left and right shin springs
    idx = [15, 29]
    spring_error = 100 * np.sum( (qpos[idx] - ref_pos[idx]) ** 2 )

    reward = 0.5 * np.exp(-joint_error) +       \
                0.3 * np.exp(-com_error) +         \
                0.1 * np.exp(-orientation_error) + \
                0.1 * np.exp(-spring_error)

    # reward = np.sign(qvel[0])*qvel[0]**2
    # desired_speed = 3.0
    # speed_diff = np.abs(qvel[0] - desired_speed)
    # if speed_diff > 1:
    #     speed_diff = speed_diff**2
    # reward = 20 - speed_diff

    return reward