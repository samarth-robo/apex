import numpy as np

def iros_paper_reward(self):
    qpos = np.copy(self.sim.qpos())

    ref_pos = self.get_ref_state(self.phase)

    # TODO: should be variable; where do these come from?
    # TODO: see magnitude of state variables to gauge contribution to reward
    weight = [0.15, 0.15, 0.1, 0.05, 0.05, 0.15, 0.15, 0.1, 0.05, 0.05]

    joint_error       = 0
    com_error         = 0
    orientation_error = 0
    spring_error      = 0

    # each joint pos
    for i, j in enumerate(self.pos_idx):
        target = ref_pos[j]
        actual = qpos[j]

        joint_error += 40 * weight[i] * (target - actual) ** 2

    # center of mass: x, y, z
    idx = [0, 1, 2]
    for j in idx:
        target = ref_pos[j]
        actual = qpos[j]

        # NOTE: in Xie et al y target is 0

        com_error += 30 * (target - actual) ** 2

    # COM orientation: qx, qy, qz
    orientation_error = 1 - np.inner(ref_pos[3:7], qpos[3:7])**2

    # left and right shin springs
    for i in [15, 29]:
        target = ref_pos[i] # NOTE: in Xie et al spring target is 0
        actual = qpos[i]

        spring_error += 1000 * (target - actual) ** 2      

    reward = 0.4 * np.exp(-joint_error) +       \
                0.4 * np.exp(-com_error) +         \
                0.1 * np.exp(-orientation_error) + \
                0.1 * np.exp(-spring_error)

    if self.debug:
        print(
            f"reward:  \t{reward:.3f} / 1.000\n"
            f"phase:  \t{self.phase/self.phaselen:.3f}\n"
            f"joint reward:\t{0.4*np.exp(-joint_error):.3f} / 0.4\n"
            f"com reward:\t{0.4*np.exp(-com_error):.3f} / 0.4\n"
            f"orientation reward:\t{0.1*np.exp(-orientation_error):.3f} / 0.1\n"
            f"spring reward:\t{0.1*np.exp(-spring_error):.3f} / 0.1\n"
        )

    # reward = np.sign(qvel[0])*qvel[0]**2
    # desired_speed = 3.0
    # speed_diff = np.abs(qvel[0] - desired_speed)
    # if speed_diff > 1:
    #     speed_diff = speed_diff**2
    # reward = 20 - speed_diff

    return reward