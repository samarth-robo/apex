import numpy as np
from cassie.cassie_keyframe import CassieKeyframeEnv
from dtw import dtw

class KeyFrameReward(object):
    def __init__(self):
        self.R = []

    def __call__(self, sim: CassieKeyframeEnv, reward_fn):
        self.R.append([reward_fn(sim, ref_state=r, free_com=True)
                       for r in sim.keyframes])
        if (sim.aslip_traj and sim.phase == sim.phaselen-1) or \
                (sim.phase == sim.phaselen):
            return self._dtw_reward(sim)
        else:
            return 0

    def _dtw_reward(self, sim: CassieKeyframeEnv):
        R = np.asarray(R)
        # reference = executed trajectory, query = keyframes
        # asymmetric = each query point is matched exactly once
        # cost matrix shape is query x reference
        alignment = dtw(-R.T, None, step_pattern='asymmetric', open_end=True,
                        open_begin=True, distance_only=True)
        reward = -alignment.distance
        self.R = []
        return reward