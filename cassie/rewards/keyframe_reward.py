import numpy as np

class KeyFrameReward(object):
    def __init__(self):
        self.R = []

    def __call__(self, sim, reward_fn):
        self.R.append([reward_fn(sim, ref_state=r, free_com=True)
                       for r in sim.keyframes])
        if (sim.aslip_traj and sim.phase == sim.phaselen-1) or \
                (sim.phase == sim.phaselen):
            return self._dtw_reward(sim)
        else:
            return 0

    def _dtw_reward(self, sim):
        # rows = keyframes, cols = trajectory
        D = -np.ascontiguousarray(np.asarray(self.R).T)
        self.R = []
        return calc_dtw(D)


def calc_dtw(D, return_match=False):
    I, J = D.shape
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
    if return_match:
        i, j = I-1, J-1
        match = [(i, j)]
        while True:
            i, j = match_i[i, j], match_j[i, j]
            match.append((i, j))
            if i == 0:
                break
        return -C[-1, -1], match
    else:
        return -C[-1, -1]