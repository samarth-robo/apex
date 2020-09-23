import numpy as np
from cassie.rewards.iros_paper_reward import iros_paper_reward

class KeyFrameReward(object):
    def __init__(self, phase_frac=0.5):
        """
        phase_frac = min fraction of phaselen in which all keyframes can be hit
        """
        self.R = []
        self.phase_frac = phase_frac

    def __call__(self, env):
        self.R.append([iros_paper_reward(env, ref_state=k) for k in env.keyframes])
        reward = (self.phase_frac/env.phaselen) * self._stability_reward(env)
        if (env.aslip_traj and env.phase == env.phaselen) or env.done or \
                (env.phase > env.phaselen) or (env.phase_state == 3):
            reward = reward + self._keyframe_reward(env) - \
                (float(env.phase)/env.phaselen)
        return reward

    def _keyframe_reward(self, env):
        """
        Matches sparse keyframes using Dynamic Time Warping, and calculates 
        maximum possible keyframe matching envilarity reward
        """
        # rows = keyframes, cols = trajectory
        D = -np.ascontiguousarray(np.asarray(self.R).T)
        self.R = []
        return calc_dtw(D)

    def _stability_reward(self, env):
        # pelvis orientation
        pelvis_orientation_cost = 1 - \
            np.inner([1, 0, 0, 0], env.cassie_state.pelvis.orientation[:])**2
        # COM sideways deviation from center of foot pos
        foot_center = (env.l_foot_pos + env.r_foot_pos) * 0.5
        com_deviation = np.linalg.norm([foot_center[0]-env.sim.qpos[0],
                                        foot_center[1]-env.sim.qpos[1]])
        # TODO: penalize leg plane not being vertical
        reward = 0.4*np.exp(-pelvis_orientation_cost) +\
            0.4*np.exp(-com_deviation) +\
            0.1*np.exp(-env.torque_cost) +\
            0.1*np.exp(-env.smooth_cost)
        return 0


def calc_dtw(D, return_match=False):
    I, J = D.shape
    small_traj = I > J
    if small_traj:
      D = D.T
      I, J = J, I
    inf = np.finfo(D.dtype).max
    C = np.zeros((I, J), dtype=D.dtype)
    C[0, 0] = D[0, 0]
    C[0, 1:] = inf
    C[1:, 0] = inf
    match_i = np.zeros(C.shape, dtype=int)
    match_j = np.zeros(C.shape, dtype=int)
    for i in range(1, I):
        C[i, :i] = inf
        C[i, i] = D[i, i] + C[i-1, i-1]
        match_i[i, :] = i-1
        match_j[i, i] = i-1
        for j in range(i+1, J):
            j_idx = np.argmin(C[i-1, i-1:j]) + i-1
            match_j[i, j] = j_idx
            C[i, j] = D[i, j] + C[i-1, j_idx]
    # CC = np.copy(C)
    # CC[CC == inf] = -1
    # plt.figure();
    # plt.imshow(CC, cmap='Reds')
    # plt.title('C')
    reward = -C[-1, -1] / I
    if small_traj:
      reward = reward / J
    if return_match:
        i, j = I-1, J-1
        match = [(i, j)]
        while True:
            i, j = match_i[i, j], match_j[i, j]
            match.append((i, j))
            if i == 0:
                break
        return reward, match
    else:
        return reward
