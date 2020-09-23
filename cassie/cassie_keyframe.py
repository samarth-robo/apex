# Consolidated Cassie environment.

from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

from .trajectory import *
from cassie.quaternion_function import *
from cassie.phase_function import *
from .rewards import *
keyframe_reward = KeyFrameReward()

from math import floor

import numpy as np
import os
import random
import copy
from functools import partial

import pickle

import torch

# Load clock based reward functions from file
def load_reward_clock_funcs(path):
    with open(path, "rb") as f:
        clock_funcs = pickle.load(f)
    return clock_funcs


class CassieKeyframeEnv:
    def __init__(self, traj='jumping', simrate=50, command_profile="traj", input_profile="full", dynamics_randomization=True,
                 learn_gains=False, reward="iros_paper:keyframes",
                 no_delta=True, ik_baseline=False,
                 config="./cassie/cassiemujoco/cassie.xml", history=0, **kwargs):

        dirname = os.path.dirname(__file__)
        self.config = config
        self.sim = CassieSim(self.config)
        # self.sim = CassieSim("./cassie/cassiemujoco/cassie_drop_step.xml")
        self.vis = None

        # legacy env attributes clock_based and phase_based
        if command_profile == "traj":
            self.clock_based, self.phase_based = False, False
        else:
            raise NotImplementedError
        if input_profile == "full":
            self.min_input = False
        elif input_profile == "min":
            self.min_input = True

        # Arguments for the simulation and state space
        self.dynamics_randomization = dynamics_randomization
        self.no_delta = no_delta
        self.ik_baseline = ik_baseline

        # Arguments for reward function
        self.reward_func = reward
        self.early_term_cutoff = -99.0 if 'keyframes' in self.reward_func else 0.3

        # CONFIGURE REF TRAJECTORY
        if traj == "aslip":
            self.speeds = np.array([x / 10 for x in range(0, 21)])
            self.trajectories = getAllTrajectories(self.speeds)
            self.num_speeds = len(self.trajectories)
            self.traj_idx = 0
            self.speed = self.speeds[self.traj_idx]
            self.trajectory = self.trajectories[self.traj_idx]
            self.aslip_traj = True
        else:
            self.aslip_traj = False
            if traj == "walking":
                traj_path = os.path.join(dirname, "trajectory", "stepdata.bin")
            elif traj == "stepping":
                traj_path = os.path.join(dirname, "trajectory", "more-poses-trial.bin")
            elif traj == "jumping":
                traj_path = os.path.join(dirname, "trajectory", "jumping.bin")
            self.trajectory = CassieTrajectory(traj_path)
            self.speed = 0

        self.observation_space, self.clock_inds, self.mirrored_obs = self.set_up_state_space()

        # Adds option for state history for FF nets
        self._obs = len(self.observation_space)
        self.history = history

        self.observation_space = np.zeros(self._obs + self._obs * self.history)

        self.P = np.array([100,  100,  88,  96,  50]) 
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        # learn gains means there is a delta on the default PD gains ***FOR EACH LEG***
        self.learn_gains = learn_gains
        if self.learn_gains:
            self.action_space = np.zeros(10 + 20)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4,
                                  -15, -16, 17, 18, 19, -10, -11, 12, 13, 14,
                                  -25, -26, 27, 28, 29, -20, -21, 22, 23, 24]
        else:
            self.action_space = np.zeros(10)
            self.mirrored_acts = [-5, -6, 7, 8, 9, -0.1, -1, 2, 3, 4]

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()
        self.simrate = simrate  # simulate X mujoco steps with same pd target. 50 brings simulation from 2000Hz to exactly 40Hz
        self.simsteps = 0       # tracks number of mujoco simulation steps
        self.time    = 0        # number of time steps in current episode
        self.phase   = 0        # portion of the phase the robot is in
        self.counter = 0        # number of phase cycles completed in episode
        self.done = False

        # NOTE: a reference trajectory represents ONE phase cycle

        # should be floor(len(traj) / simrate) - 1
        # should be VERY cautious here because wrapping around trajectory
        # badly can cause assymetrical/bad gaits
        self.keyframes = []
        if self.aslip_traj:
            self.phaselen = self.trajectory.length - 1
        elif 'keyframes' not in self.reward_func:
            self.phaselen = floor(len(self.trajectory) / self.simrate) - 1
        else:
            self.phaselen = 2500  # maximum phase length
            self.keyframes = [(self.trajectory.qpos[i], self.trajectory.qvel[i])
                for i in range(len(self.trajectory))]
        self.phase_add = 1

        # keeps track of progress:
        # 0: start
        # 1: no feet contacting the ground
        # 2: both feet contacting the ground after being up
        # 3: some time passes after phase_state 2
        self.phase_state = 0
        self.landing_phase = 0

        # NOTE: phase_based modifies self.phaselen throughout training

        # Set up phase based / load in clock based reward func
        self.strict_relaxer = 0.1
        self.have_incentive = False if "no_incentive" in self.reward_func else True
        self.early_reward = False
        if self.phase_based:
            self.set_up_phase_reward()
        elif self.clock_based:
            self.set_up_clock_reward(dirname)

        # see include/cassiemujoco.h for meaning of these indices
        self.pos_idx = [7, 8, 9, 14, 20, 21, 22, 23, 28, 34]
        self.vel_idx = [6, 7, 8, 12, 18, 19, 20, 21, 25, 31]

        self.pos_index = np.array([1,2,3,4,5,6,7,8,9,14,15,16,20,21,22,23,28,29,30,34])
        self.vel_index = np.array([0,1,2,3,4,5,6,7,8,12,13,14,18,19,20,21,25,26,27,31])

        # CONFIGURE OFFSET for No Delta Policies
        if self.aslip_traj:
            ref_pos, ref_vel = self.get_ref_state(self.phase)
            self.offset = ref_pos[self.pos_idx]
        else:
            self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        self.max_orient_change = 0.2

        self.max_simrate = self.simrate + 10
        self.min_simrate = self.simrate - 20

        self.max_speed = 4.0
        self.min_speed = -0.3

        self.max_side_speed  = 0.3
        self.min_side_speed  = -0.3

        # global flat foot orientation, can be useful part of reward function:
        self.neutral_foot_orient = np.array([-0.24790886454547323, -0.24679713195445646, -0.6609396704367185, 0.663921021343526])
        
        # various tracking variables for reward funcs
        self.stepcount = 0
        self.l_high = False  # only true if foot is above 0.2m 
        self.r_high = False
        self.l_swing = False  # these will be true even if foot is barely above ground
        self.r_swing = False
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_vel = np.zeros(3)
        self.r_foot_vel = np.zeros(3)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0
        self.torque_cost = 0
        self.smooth_cost = 0

        # TODO: should this be mujoco tracking var or use state estimator. real command interface will use state est
        # Track pelvis position as baseline for pelvis tracking command inputs
        self.last_pelvis_pos = self.sim.qpos()[0:3]

        #### Dynamics Randomization ####
        self.dynamics_randomization = dynamics_randomization
        self.slope_rand = dynamics_randomization
        self.joint_rand = dynamics_randomization

        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03
        
        self.encoder_noise = 0.01
        
        self.damping_low = 0.3
        self.damping_high = 5.0

        self.mass_low = 0.5
        self.mass_high = 1.5

        self.fric_low = 0.4
        self.fric_high = 1.1

        self.speed = 0
        self.side_speed = 0
        self.orient_add = 0

        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_geom_friction()
        self.default_rgba = self.sim.get_geom_rgba()
        self.default_quat = self.sim.get_geom_quat()

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

        # Keep track of actions, torques
        self.prev_action = None
        self.curr_action = None
        self.prev_torque = None

        self.debug = False

    # Set up phase reward for dynamic phase functions
    def set_up_phase_reward(self):

        if "early" in self.reward_func:
            self.early_reward = True

        if "library" in self.reward_func:
            self.phase_input_mode = "library"
        else:
            self.phase_input_mode = None  # Vary all parts of input

        if "no_speed" in self.reward_func:
            self.reward_func = "no_speed_clock"
        else:
            self.reward_func = "clock"

    # Set up clock reward for loaded in phase functions
    def set_up_clock_reward(self, dirname):
        
        if "early" in self.reward_func:
            self.early_reward = True

        if "load" in self.reward_func:
            self.reward_clock_func = load_reward_clock_funcs(os.path.join(dirname, "rewards", "reward_clock_funcs", self.reward_func + ".pkl"))
            self.left_clock = self.reward_clock_func["left"]
            self.right_clock = self.reward_clock_func["right"]
            self.reward_func = "load_clock"
        elif "aslip" in self.reward_func:
            self.reward_clock_func = load_reward_clock_funcs(os.path.join(dirname, "rewards", "reward_clock_funcs", "no_incentive_aslip_clock_strict0.3.pkl"))
            self.left_clock = self.reward_clock_func["left"]
            self.right_clock = self.reward_clock_func["right"]
            self.reward_func = "aslip_clock"
        else:

            if "grounded" in self.reward_func:
                self.stance_mode = "grounded"
            elif "aerial" in self.reward_func:
                self.stance_mode = "aerial"
            else:
                self.stance_mode = "zero"

            # Clock based rewards are organized as follows: "<approach>_<pkl-file-path>" where <approach> is either "" or "max_vel" or "switch"
            # extract <approach> from reward string
            if "max_vel" in self.reward_func:    
                self.reward_func = "max_vel_clock"
            elif "switch" in self.reward_func:
                # switch from walking to running, where walking path is specified
                self.switch_speed = 1.8
                self.reward_func = "clock"
            else:
                # match commanded speed input. maybe constrained to walking or running or figuring out based on specified path
                # approach = ""
                self.reward_func = "clock"

    def set_up_state_space(self):

        full_state_est_size = 46
        min_state_est_size = 21
        speed_size     = 2      # x speed, y speed
        clock_size     = 2      # sin, cos
        phase_size     = 5      # swing duration, stance duration, one-hot encoding of stance mode

        # input --> FULL
        if not self.min_input:
            base_mir_obs = np.array([0.1, 1, -2, 3, -4, -10, -11, 12, 13, 14, -5, -6, 7, 8, 9, 15, -16, 17, -18, 19, -20, -26, -27, 28, 29, 30, -21, -22, 23, 24, 25, 31, -32, 33, 37, 38, 39, 34, 35, 36, 43, 44, 45, 40, 41, 42])
            obs_size = full_state_est_size

        # input --> MIN
        else:
            base_mir_obs = np.array([
                3, 4, 5,            # L foot relative pos
                0.1, 1, 2,          # R foot relative pos
                6, -7, 8, -9,       # pelvis orient (quaternion)
                -10, 11, -12,       # pelvis rot Vel
                17, -18, 19, -20,   # L foot orient
                13, -14, 15, -16    # R foot orient
            ])
            obs_size = min_state_est_size
        
        # command --> CLOCK_BASED : clock, speed
        if self.clock_based:
            append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+speed_size)])
            mirrored_obs = np.concatenate([base_mir_obs, append_obs])
            clock_inds = append_obs[0:clock_size].tolist()
            obs_size += clock_size + speed_size

        # command --> PHASE_BASED : clock, phase info, speed
        elif self.phase_based:
            append_obs = np.array([len(base_mir_obs) + i for i in range(clock_size+phase_size+speed_size)])
            mirrored_obs = np.concatenate([base_mir_obs, append_obs])
            clock_inds = append_obs[0:clock_size].tolist()
            obs_size += clock_size + phase_size + speed_size

        # command --> REF_TRAJ_BASED : single-timestep trajectory
        else:
            # Find the mirrored trajectory
            if self.aslip_traj:
                ref_traj_size = 18
                mirrored_traj = np.array([6,7,8,9,10,11,0.1,1,2,3,4,5,12,13,14,15,16,17])
            else:
                ref_traj_size = 40
                mirrored_traj = np.array([0.1, 1, 2, 3, 4, 5, -13, -14, 15, 16, 17, 18, 19, -6, -7, 8, 9, 10, 11, 12,
                                          20, 21, 22, 23, 24, 25, -33, -34, 35, 36, 37, 38, 39, -26, -27, 28, 29, 30, 31, 32])
            mirrored_traj_sign = np.multiply(np.sign(mirrored_traj), obs_size+np.floor(np.abs(mirrored_traj)))
            mirrored_obs = np.concatenate([base_mir_obs, mirrored_traj_sign])
            clock_inds = None
            obs_size += ref_traj_size

        observation_space = np.zeros(obs_size)
        mirrored_obs = mirrored_obs.tolist()

        return observation_space, clock_inds, mirrored_obs

    def rotate_to_orient(self, vec):
        quaternion  = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def step_simulation(self, action, learned_gains=None):

        if not self.ik_baseline:
            if self.aslip_traj and self.phase == self.phaselen - 1:
                ref_pos, ref_vel = self.get_ref_state(0)
            else:
                ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)
        else:
            ref_pos = self.trajectory.ik_pos[self.simsteps]

        if not self.no_delta:
            self.offset = ref_pos[self.pos_idx]
        target = action + self.offset

        if self.joint_rand:
            target -= self.motor_encoder_noise

        foot_pos = np.zeros(6)
        self.sim.foot_pos(foot_pos)
        prev_foot = copy.deepcopy(foot_pos)
        self.u = pd_in_t()
        for i in range(5):

            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            if not self.learn_gains:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i]
            else:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + learned_gains[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i] + learned_gains[5+i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + learned_gains[10+i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i] + learned_gains[15+i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)
        self.sim.foot_pos(foot_pos)
        self.l_foot_vel = (foot_pos[0:3] - prev_foot[0:3]) / 0.0005
        self.r_foot_vel = (foot_pos[3:6] - prev_foot[3:6]) / 0.0005
        foot_forces = self.sim.get_foot_forces()
        if self.l_high and foot_forces[0] > 0:
            self.l_high = False
            self.stepcount += 1
        elif not self.l_high and foot_pos[2] >= 0.2:
            self.l_high = True
        if self.r_high and foot_forces[0] > 0:
            self.stepcount += 1
            self.r_high = False
        elif not self.r_high and foot_pos[5] >= 0.2:
            self.r_high = True

        if self.l_swing and foot_forces[0] > 0:
            self.l_swing = False
        elif not self.l_swing and foot_pos[2] >= 0:
            self.l_swing = True
        if self.r_swing and foot_forces[0] > 0:
            self.r_swing = False
        elif not self.r_swing and foot_pos[5] >= 0:
            self.r_swing = True

    # Basic version of step_simulation, that only simulates forward in time, does not do any other
    # computation for reward, etc. Is faster and should be used for evaluation purposes
    def step_sim_basic(self, action, learned_gains=None):
        if not self.ik_baseline:
            if self.aslip_traj and self.phase == self.phaselen - 1:
                ref_pos, ref_vel = self.get_ref_state(0)
            else:
                ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)
        else:
            ref_pos = self.trajectory.ik_pos[self.simsteps]

        if not self.no_delta:
            self.offset = ref_pos[self.pos_idx]
        target = action + self.offset

        if self.joint_rand:
            target -= self.motor_encoder_noise

        self.u = pd_in_t()
        for i in range(5):

            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            if self.learn_gains is False:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i]
            else:
                self.u.leftLeg.motorPd.pGain[i]  = self.P[i] + learned_gains[i]
                self.u.rightLeg.motorPd.pGain[i] = self.P[i] + learned_gains[5+i]
                self.u.leftLeg.motorPd.dGain[i]  = self.D[i] + learned_gains[10+i]
                self.u.rightLeg.motorPd.dGain[i] = self.D[i] + learned_gains[15+i]

            self.u.leftLeg.motorPd.torque[i]  = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i]  = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i]  = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)

    def step(self, action, return_omniscient_state=False, f_term=0):
        
        if self.dynamics_randomization:
            simrate = np.random.uniform(self.max_simrate, self.min_simrate)
        else:
            simrate = self.simrate

        # reset mujoco tracking variables
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        foot_pos = np.zeros(6)
        self.l_foot_pos = np.zeros(3)
        self.r_foot_pos = np.zeros(3)
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0
        self.smooth_cost = 0
        self.torque_cost = 0

        if self.learn_gains:
            action, learned_gains = action[0:10], action[10:]

        for i in range(self.simrate):
            if self.learn_gains:
                self.step_simulation(action, learned_gains)
            else:
                self.step_simulation(action)
            self.simsteps += 1
            qpos = np.copy(self.sim.qpos())
            qvel = np.copy(self.sim.qvel())
            # Foot Force Tracking
            foot_forces = self.sim.get_foot_forces()
            self.l_foot_frc += foot_forces[0]
            self.r_foot_frc += foot_forces[1]
            # Relative Foot Position tracking
            self.sim.foot_pos(foot_pos)
            self.l_foot_pos = foot_pos[0:3]
            self.r_foot_pos = foot_pos[3:6]
            # Foot Orientation Cost
            self.l_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("left-foot")) ** 2)
            self.r_foot_orient_cost += (1 - np.inner(self.neutral_foot_orient, self.sim.xquat("right-foot")) ** 2)
            # Hip Yaw velocity cost
            self.hiproll_cost += (np.abs(qvel[6]) + np.abs(qvel[19])) / 3
            if self.prev_action is not None:
                self.hiproll_act += 2*np.linalg.norm(self.prev_action[[0, 5]] - action[[0, 5]])
            else:
                self.hiproll_act += 0
            curr_torques = np.array(self.cassie_state.motor.torque[:])
            if self.prev_torque is not None:
                self.smooth_cost += 0.0001*np.linalg.norm(np.square(curr_torques - self.prev_torque))
            else:
                self.smooth_cost += 0
            self.prev_torque = curr_torques
            self.torque_cost += 0.00006*np.linalg.norm(np.square(curr_torques))
        
        self.l_foot_frc              /= self.simrate
        self.r_foot_frc              /= self.simrate
        self.l_foot_orient_cost      /= self.simrate
        self.r_foot_orient_cost      /= self.simrate
        self.hiproll_cost            /= self.simrate
        self.hiproll_act             /= self.simrate
        self.smooth_cost             /= self.simrate
        self.torque_cost             /= self.simrate

        self.time  += 1
        self.phase += self.phase_add
        self.update_phase_state()

        # no more knee walking
        # if self.sim.xpos("left-tarsus")[2] < 0.1 or self.sim.xpos("right-tarsus")[2] < 0.1:
        #     done = True
            # print("left tarsus: {:.2f}\tleft foot: {:.2f}".format(self.sim.xpos("left-tarsus")[2], self.sim.xpos("left-foot")[2]))
            # print("right tarsus: {:.2f}\tright foot: {:.2f}".format(self.sim.xpos("right-tarsus")[2], self.sim.xpos("right-foot")[2]))
            # while(1):
            #     self.vis.draw(self.sim)
        self.done = False
        # TODO: make 0.3 a variable/more transparent
        height = self.sim.qpos()[2]
        if (height < 0.4) or (height > 5.0):
            self.done = True

        # make sure trackers aren't None and calculate reward
        self.curr_action = action
        if self.prev_action is None:
            self.prev_action = action
        if self.prev_torque is None:
            self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])
        reward = self.compute_reward(action)
        if reward < self.early_term_cutoff:
            self.done = True

        # update previous action
        self.prev_action = action
        # update previous torque
        self.prev_torque = np.asarray(self.cassie_state.motor.torque[:])

        if (self.aslip_traj and self.phase >= self.phaselen) or \
                (self.phase > self.phaselen) or (self.phase_state == 3):
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.simsteps = 0
            self.phase = 0
            self.counter += 1
            self.phase_state = 0
            self.landing_phase = 0

        if np.random.randint(300) == 0:  # random changes to orientation
            self.orient_add += np.random.uniform(-self.max_orient_change,
                                                 self.max_orient_change)

        if np.random.randint(100) == 0:  # random changes to speed
            self.speed = np.random.uniform(self.min_speed, self.max_speed)
            self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
        
        if np.random.randint(300) == 0:  # random changes to sidespeed
            self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state(), reward, \
                self.done, {}
        else:
            return self.get_full_state(), reward, self.done, {}

    # More basic, faster version of step
    def step_basic(self, action, return_omniscient_state=False):

        if self.learn_gains:
            action, learned_gains = action[0:10], action[10:]

        for i in range(self.simrate):
            if self.learn_gains:
                self.step_sim_basic(action, learned_gains)
            else:
                self.step_sim_basic(action)
            self.simsteps += 1

        self.time  += 1
        self.phase += self.phase_add
        self.update_phase_state()

        if (self.aslip_traj and self.phase >= self.phaselen) or \
                (self.phase > self.phaselen) or (self.phase_state == 3):
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.simsteps = 0
            self.phase = 0
            self.counter += 1
            self.phase_state = 0
            self.landing_phase = 0

        if return_omniscient_state:
            return self.get_full_state(), self.get_omniscient_state()
        else:
            return self.get_full_state()

    def update_phase_state(self):
        if self.l_high and self.r_high:
            self.phase_state = 1
        elif not (self.l_high or self.r_high):
            if self.phase_state == 1:
                self.phase_state = 2
                self.landing_phase = self.phase
            elif (self.phase_state == 2) and \
                    (self.phase > self.landing_phase + 5):
                self.phase_state = 3
                self.landing_phase = 0

    def reset(self):

        if self.aslip_traj:
            self.traj_idx = random.randint(0, self.num_speeds-1)
            self.speed = self.speeds[self.traj_idx]
            # print("current speed: {}\tcurrent traj: {}".format(self.speed, random_speed_idx))
            self.trajectory = self.trajectories[self.traj_idx]  # switch the current trajectory
            self.phaselen = self.trajectory.length - 1
        else:
            self.speed = (random.randint(0, 40)) / 10
            self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
            # # Make sure that if speed is above 2, freq is at least 1.2
            # if self.speed > 1.3:# or np.any(self.speed_schedule > 1.6):
            #     self.phase_add = 1.3 + 0.7*random.random()
            # else:
            #     self.phase_add = 1 + random.random()

        # Set up phase based
        if self.phase_based:
            if self.phase_input_mode == "library":
                # constrain speed a bit further
                self.speed = (random.randint(0, 30)) / 10
                # library input -- pick total duration and swing/stance phase based on that. also randomize grounded vs aerial vs zero
                total_duration = random.randint(3, 6) / 10  # this if for one swing and stance, so 2 * total_duration = walk cycle time
                ratio = random.randint(2, 8) / 10
                self.swing_duration = total_duration * ratio
                self.stance_duration = total_duration - self.swing_duration
                # also change stance mode
                self.stance_mode = np.random.choice(["grounded", "aerial", "zero"])
            else:
                # random everything
                self.swing_duration = random.randint(1, 50) / 100
                self.stance_duration = random.randint(1, 30) / 100
                self.stance_mode = np.random.choice(["grounded", "aerial", "zero"])
            self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration, self.stance_duration, self.strict_relaxer, self.stance_mode, self.have_incentive, FREQ=2000//self.simrate)
        # ELSE load in clock based reward func
        elif self.reward_func == "aslip_clock" or self.reward_func == "load_clock":
            pass
        # ELSE use simple relationship to define swing and stance duration
        elif self.clock_based:
            if self.reward_func == "switch_clock":
                if self.speed < self.switch_speed:
                    self.stance_mode = "grounded"
                else:
                    self.stance_mode = "aerial"
            total_duration = (0.9 - 0.25 / 3.0 * abs(self.speed)) / 2
            self.swing_duration = (0.30 + ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
            self.stance_duration = (0.70 - ((0.70 - 0.30) / 3) * abs(self.speed)) * total_duration
            self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration, self.stance_duration, self.strict_relaxer, self.stance_mode, self.have_incentive, FREQ=2000//self.simrate)

        self.simsteps = 0

        self.phase = 0
        self.time = 0
        self.counter = 0
        self.phase_state = 0
        self.landing_phase = 0
        self.done = False

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]],
                                [damp[1], damp[1]],
                                [damp[2], damp[2]],
                                [damp[3], damp[3]],
                                [damp[4], damp[4]],
                                [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6]*self.damping_low, damp[6]*self.damping_high],
                              [damp[7]*self.damping_low, damp[7]*self.damping_high],
                              [damp[8]*self.damping_low, damp[8]*self.damping_high]]          # 6->8 and 19->21

            achilles_damp_range = [[damp[9]*self.damping_low, damp[9]*self.damping_high],
                                   [damp[10]*self.damping_low, damp[10]*self.damping_high],
                                   [damp[11]*self.damping_low, damp[11]*self.damping_high]]   # 9->11 and 22->24

            knee_damp_range     = [[damp[12]*self.damping_low, damp[12]*self.damping_high]]   # 12 and 25
            shin_damp_range     = [[damp[13]*self.damping_low, damp[13]*self.damping_high]]   # 13 and 26
            tarsus_damp_range   = [[damp[14]*self.damping_low, damp[14]*self.damping_high]]   # 14 and 27

            heel_damp_range     = [[damp[15], damp[15]]]                                      # 15 and 28
            fcrank_damp_range   = [[damp[16]*self.damping_low, damp[16]*self.damping_high]]   # 16 and 29
            prod_damp_range     = [[damp[17], damp[17]]]                                      # 17 and 30
            foot_damp_range     = [[damp[18]*self.damping_low, damp[18]*self.damping_high]]   # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range      = [[self.mass_low*m[1], self.mass_high*m[1]]]   # 1
            hip_mass_range         = [[self.mass_low*m[2], self.mass_high*m[2]],   # 2->4 and 14->16
                                    [self.mass_low*m[3], self.mass_high*m[3]],
                                    [self.mass_low*m[4], self.mass_high*m[4]]]

            achilles_mass_range    = [[self.mass_low*m[5], self.mass_high*m[5]]]    # 5 and 17
            knee_mass_range        = [[self.mass_low*m[6], self.mass_high*m[6]]]    # 6 and 18
            knee_spring_mass_range = [[self.mass_low*m[7], self.mass_high*m[7]]]    # 7 and 19
            shin_mass_range        = [[self.mass_low*m[8], self.mass_high*m[8]]]    # 8 and 20
            tarsus_mass_range      = [[self.mass_low*m[9], self.mass_high*m[9]]]    # 9 and 21
            heel_spring_mass_range = [[self.mass_low*m[10], self.mass_high*m[10]]]  # 10 and 22
            fcrank_mass_range      = [[self.mass_low*m[11], self.mass_high*m[11]]]  # 11 and 23
            prod_mass_range        = [[self.mass_low*m[12], self.mass_high*m[12]]]  # 12 and 24
            foot_mass_range        = [[self.mass_low*m[13], self.mass_high*m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric)/3)):
                fric_noise += [translational, torsional, rolling]

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
        else:
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_geom_friction(self.default_fric)

        if self.slope_rand:
            geom_plane = [np.random.uniform(-self.max_roll_incline, self.max_roll_incline), np.random.uniform(-self.max_pitch_incline, self.max_pitch_incline), 0]
            quat_plane   = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat  = list(quat_plane) + list(self.default_quat[4:])
            self.sim.set_geom_quat(geom_quat)
        else:
            self.sim.set_geom_quat(self.default_quat)

        if self.joint_rand:
            self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
            self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)
        else:
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        # apply dynamics
        self.sim.set_const()

        qpos, qvel = self.get_ref_state(self.phase)

        if self.aslip_traj:
            qvel = np.zeros(qvel.shape)

        self.sim.set_qpos(qpos)
        self.sim.set_qvel(qvel)

        self.last_pelvis_pos = self.sim.qpos()[0:3]

        # Need to reset u? Or better way to reset cassie_state than taking step
        self.cassie_state = self.sim.step_pd(self.u)

        # reset commands
        self.orient_add = 0  # random.randint(-10, 10) * np.pi / 25
        self.speed = np.random.uniform(self.min_speed, self.max_speed)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        # reset mujoco tracking variables
        self.l_foot_frc = 0
        self.r_foot_frc = 0
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.hiproll_cost = 0
        self.hiproll_act = 0
        self.torque_cost = 0
        self.smooth_cost = 0
        self.prev_action = None
        self.prev_torque = None

        return self.get_full_state()

    def reset_for_test(self, full_reset=False):
        self.simsteps = 0
        self.phase = 0
        self.time = 0
        self.counter = 0
        self.orient_add = 0
        self.phase_add = 1
        self.phase_state = 0
        self.landing_phase = 0
        self.done = False

        self.state_history = [np.zeros(self._obs) for _ in range(self.history+1)]

        if self.aslip_traj:
            self.traj_idx = 0
            self.speed = 0
            # print("current speed: {}".format(self.speed))
            self.trajectory = self.trajectories[self.traj_idx]  # switch the current trajectory
            self.phaselen = self.trajectory.length - 1
        else:
            self.speed = 0

        # load in clock based reward func
        if self.reward_func == "aslip_clock" or self.reward_func == "load_clock":
            pass
        # ELSE use simple relationship to define swing and stance duration
        else:
            # variable inputs
            self.swing_duration = 0.15
            self.stance_duration = 0.25
            self.stance_mode = "grounded"
            self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration, self.stance_duration, self.strict_relaxer, self.stance_mode, self.have_incentive, FREQ=2000//self.simrate)

        if not full_reset:
            qpos, qvel = self.get_ref_state(self.phase)
            self.sim.set_qpos(qpos)
            self.sim.set_qvel(qvel)

            # reset mujoco tracking variables
            self.last_pelvis_pos = self.sim.qpos()[0:3]
            self.l_foot_frc = 0
            self.r_foot_frc = 0

            # Need to reset u? Or better way to reset cassie_state than taking step
            self.cassie_state = self.sim.step_pd(self.u)
        else:
            self.sim.full_reset()
            self.reset_cassie_state()

        if self.dynamics_randomization:
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_const()

        if self.slope_rand:
            self.sim.set_geom_quat(np.array([1, 0, 0, 0]), "floor")

        if self.joint_rand:
            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        # reward terms
        self.l_foot_orient_cost = 0
        self.r_foot_orient_cost = 0
        self.smooth_cost = 0
        self.torque_cost = 0
        self.hiproll_act = 0
        self.hiproll_cost = 0
        self.prev_action = None
        self.prev_torque = None

        return self.get_full_state()

    def reset_cassie_state(self):
        # Only reset parts of cassie_state that is used in get_full_state
        self.cassie_state.pelvis.position[:] = [0, 0, 1.01]
        self.cassie_state.pelvis.orientation[:] = [1, 0, 0, 0]
        self.cassie_state.pelvis.rotationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalVelocity[:] = np.zeros(3)
        self.cassie_state.pelvis.translationalAcceleration[:] = np.zeros(3)
        self.cassie_state.terrain.height = 0
        self.cassie_state.motor.position[:] = [0.0045, 0, 0.4973, -1.1997, -1.5968, 0.0045, 0, 0.4973, -1.1997, -1.5968]
        self.cassie_state.motor.velocity[:] = np.zeros(10)
        self.cassie_state.joint.position[:] = [0, 1.4267, -1.5968, 0, 1.4267, -1.5968]
        self.cassie_state.joint.velocity[:] = np.zeros(6)

    # Helper function for updating the speed, used in visualization tests
    # not needed in training cause we don't change speeds in middle of rollout, and
    # we randomize the starting phase of each rollout
    def update_speed(self, new_speed, new_side_speed=0.0):

        if self.aslip_traj:
            self.traj_idx = (np.abs(self.speeds - new_speed)).argmin()
            self.speed = self.traj_idx / 10
            self.trajectory = self.trajectories[self.traj_idx]
            old_phaselen = self.phaselen
            self.phaselen = self.trajectory.length - 1
            # update phase
            self.phase = int(self.phaselen * self.phase / old_phaselen)
            # new offset
            ref_pos, ref_vel = self.get_ref_state(self.phase)
            self.offset = ref_pos[self.pos_idx]
        else:
            self.speed = np.clip(new_speed, self.min_speed, self.max_speed)
        
        self.side_speed = np.clip(new_side_speed, self.min_side_speed, self.max_side_speed)

        if self.phase_based:
            self.swing_duration = max(0.01, self.swing_duration)
            self.stance_duration = max(0.01, self.stance_duration)
            old_phaselen = self.phaselen
            self.set_up_phase_reward()
            self.phase = int(self.phaselen * self.phase / old_phaselen)
        elif self.reward_func == "aslip_clock":
            pass
        else:
            total_duration = (0.9 - 0.25 / 3.0 * self.speed) / 2
            self.swing_duration = (0.30 + ((0.70 - 0.30) / 3) * self.speed) * total_duration
            self.stance_duration = (0.70 - ((0.70 - 0.30) / 3) * self.speed) * total_duration
            old_phaselen = self.phaselen
            self.left_clock, self.right_clock, self.phaselen = create_phase_reward(self.swing_duration, self.stance_duration, self.strict_relaxer, self.stance_mode, self.have_incentive, FREQ=2000//self.simrate)
            self.phase = int(self.phaselen * self.phase / old_phaselen)

    def compute_reward(self, action):

        if self.reward_func == "clock" or self.reward_func == "load_clock" or self.reward_func == "switch_clock":
            self.early_term_cutoff = -99.
            if self.early_reward:
                return early_clock_reward(self, action)
            else:
                return clock_reward(self, action)
        elif self.reward_func == "no_speed_clock":
            self.early_term_cutoff = -99.
            return no_speed_clock_reward(self, action)
        elif self.reward_func == "max_vel_clock":
            self.early_term_cutoff = -99.
            return max_vel_clock_reward(self, action)
        elif self.reward_func == "aslip_clock":
            self.early_term_cutoff = -99.
            return aslip_clock_reward(self, action)
        elif self.reward_func == "aslip_old":
            self.early_term_cutoff = 0.0
            return aslip_old_reward(self, action)
        elif self.reward_func == "iros_paper":
            return iros_paper_reward(self)
        elif self.reward_func == "5k_speed_reward":
            return old_speed_reward(self)
        elif self.reward_func == "trajmatch_footorient_hiprollvelact_reward":
            return trajmatch_footorient_hiprollvelact_reward(self)
        elif self.reward_func == "speedmatch_footorient_hiprollvelact_reward":
            return speedmatch_footorient_hiprollvelact_reward(self)
        elif self.reward_func == "iros_paper_keyframes":
            return keyframe_reward(self)
        else:
            raise NotImplementedError

    # get the corresponding state from the reference trajectory for the current phase
    def get_ref_state(self, phase=None):
        if phase is None:
            phase = self.phase

        if phase > self.phaselen:
            phase = 0

        # TODO: make this not so hackish
        if phase > floor(len(self.trajectory) / self.simrate) - 1:
            phase = floor((phase / self.phaselen) * len(self.trajectory) / self.simrate)

        desired_ind = phase * self.simrate if not self.aslip_traj else phase
        # phase_diff = desired_ind - math.floor(desired_ind)
        # if phase_diff != 0:       # desired ind is an int
        #     pos_prev = np.copy(self.trajectory.qpos[math.floor(desired_ind)])
        #     vel_prev = np.copy(self.trajectory.qvel[math.floor(desired_ind)])
        #     pos_next = np.copy(self.trajectory.qpos[math.ceil(desired_ind)])
        #     vel_next = np.copy(self.trajectory.qvel[math.ceil(desired_ind)])
        #     pos = pos_prev + phase_diff * (pos_next - pos_prev)
        #     vel = vel_prev + phase_diff * (vel_next - vel_prev)
        # else:
        # print("desired ind: ", desired_ind)
        pos = np.copy(self.trajectory.qpos[int(desired_ind)])
        vel = np.copy(self.trajectory.qvel[int(desired_ind)])

        # this is just setting the x to where it "should" be given the number
        # of cycles
        # pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter

        # ^ should only matter for COM error calculation,
        # gets dropped out of state variable for input reasons

        ### Setting variable speed
        if not self.aslip_traj:
            pos[0] *= self.speed
            pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter * self.speed
        else:
            pos[0] += (self.trajectory.qpos[-1, 0] - self.trajectory.qpos[0, 0]) * self.counter

        # setting lateral distance target to 0?
        # regardless of reference trajectory?
        pos[1] = 0

        if not self.aslip_traj:
            vel[0] *= self.speed

        return pos, vel

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        ref_pos, ref_vel = self.get_ref_state(self.phase + self.phase_add)

        # TODO: maybe convert to set subtraction for clarity
        # {i for i in range(35)} -
        # {0, 10, 11, 12, 13, 17, 18, 19, 24, 25, 26, 27, 31, 32, 33}

        # this is everything except pelvis x and qw, achilles rod quaternions,
        # and heel spring/foot crank/plantar rod angles
        # note: x is forward dist, y is lateral dist, z is height

        # makes sense to always exclude x because it is in global coordinates and
        # irrelevant to phase-based control. Z is inherently invariant to
        # trajectory despite being global coord. Y is only invariant to straight
        # line trajectories.

        # command --> PHASE_BASED : clock, phase info, speed
        if self.phase_based:
            clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                    np.cos(2 * np.pi * self.phase / self.phaselen)]
            ext_state = np.concatenate((clock, [self.swing_duration, self.stance_duration], encode_stance_mode(self.stance_mode), [self.speed, self.side_speed]))
        # command --> CLOCK_BASED : clock, speed
        elif self.clock_based:
            clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                    np.cos(2 * np.pi * self.phase / self.phaselen)]
            ext_state = np.concatenate((clock, [self.speed, self.side_speed]))
        # command --> REF_TRAJ_BASED : aslip trajectory
        elif self.aslip_traj and not self.clock_based:
            if(self.phase == 0):
                # ext_state = np.concatenate(get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, self.phaselen - 1, offset=self.vertOffset))
                ext_state = np.concatenate(get_ref_aslip_unaltered_state(self, self.phaselen - 1))
            else:
                # ext_state = np.concatenate(get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, self.phaselen - 1, offset=self.vertOffset))
                ext_state = np.concatenate(get_ref_aslip_unaltered_state(self, self.phase))
        # command --> REF_TRAJ_BASED : agility trajectory
        else:
            ext_state = np.concatenate([ref_pos[self.pos_index], ref_vel[self.vel_index]])

        # Update orientation
        new_orient = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        new_translationalVelocity = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
        new_translationalAcceleleration = self.rotate_to_orient(self.cassie_state.pelvis.translationalAcceleration[:])
        
        # motor and joint poses
        if self.joint_rand:
            motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
            joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
        else:
            motor_pos = self.cassie_state.motor.position[:]
            joint_pos = self.cassie_state.joint.position[:]

        if self.min_input:
            robot_state = np.concatenate([
                self.cassie_state.leftFoot.position[:],             # left foot position
                self.cassie_state.rightFoot.position[:],            # right foot position
                new_orient,                                         # pelvis orientation
                self.cassie_state.pelvis.rotationalVelocity[:],     # pelvis rotational velocity
                self.cassie_state.leftFoot.orientation[:],          # left foot orientation
                self.cassie_state.rightFoot.orientation[:]          # right foot orientation
            ])
        else:
            robot_state = np.concatenate([
                [self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height],  # pelvis height
                new_orient,                                         # pelvis orientation
                motor_pos,                                          # actuated joint positions
                new_translationalVelocity,                          # pelvis translational velocity
                self.cassie_state.pelvis.rotationalVelocity[:],     # pelvis rotational velocity
                self.cassie_state.motor.velocity[:],                # actuated joint velocities
                new_translationalAcceleleration,                    # pelvis translational acceleration
                joint_pos,                                          # unactuated joint positions
                self.cassie_state.joint.velocity[:]                 # unactuated joint velocities
            ])

        state = np.concatenate([robot_state, ext_state])

        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history+1]

        return np.concatenate(self.state_history)

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, self.config)

        return self.vis.draw(self.sim)

    def get_state_info(self):
        # state estimator info
        pelvis_pos = self.cassie_state.pelvis.position[:]
        pelvis_pos[2] = pelvis_pos[2] - self.cassie_state.terrain.height
        lf_pos = self.cassie_state.leftFoot.position[:]
        rf_pos = self.cassie_state.rightFoot.position[:]
        lf_pos_global = [pelvis_pos[i] + lf_pos[i] for i in range(3)]
        rf_pos_global = [pelvis_pos[i] + rf_pos[i] for i in range(3)]
        # robot_state_info = np.array([pelvis_pos, lf_pos, rf_pos])
        robot_state_info = np.array([pelvis_pos, lf_pos_global, rf_pos_global])

        # mujoco info
        qpos = self.sim.qpos()
        actual_compos = qpos[0:3]
        actual_lf = self.l_foot_pos
        actual_rf = self.r_foot_pos
        # actual_lf = self.l_foot_pos - qpos[0:3]
        # actual_rf = self.r_foot_pos - qpos[0:3]
        actual_state_info = np.array([actual_compos, actual_lf, actual_rf])

        return robot_state_info, actual_state_info

    # This is assumed to be called after env.step. hence, using self.phase - 1 
    def get_traj_and_state_info(self):
        # traj info used in rewards
        traj_info = get_ref_aslip_global_state(self, self.phase-1)
        # traj_info = get_ref_aslip_unaltered_state(self, self.phase)
        traj_info = [traj_info[4], traj_info[2], traj_info[0]]

        # traj info going into the policy
        # traj_cmd_info = get_ref_aslip_ext_state(self, self.cassie_state, self.last_pelvis_pos, self.phase, offset=self.vertOffset)
        traj_cmd_info = get_ref_aslip_unaltered_state(self, self.phase-1)
        traj_cmd_info = [traj_cmd_info[4], traj_cmd_info[2], traj_cmd_info[0]]

        robot_state_info, actual_state_info = self.get_state_info()

        return traj_info, traj_cmd_info, robot_state_info, actual_state_info


# Currently unused
# def get_omniscient_state(self):
#     full_state = self.get_full_state()
#     omniscient_state = np.hstack((full_state, self.sim.get_dof_damping(), self.sim.get_body_mass(), self.sim.get_body_ipos(), self.sim.get_ground_friction))
#     return omniscient_state

# nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10:  left heel spring
# 12:  left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation wx
# [ 4] Pelvis orientation wy
# [ 5] Pelvis orientation wz
# [ 6] Left hip roll         (Motor [0])
# [ 7] Left hip yaw          (Motor [1])
# [ 8] Left hip pitch        (Motor [2])
# [ 9] Left knee             (Motor [3])
# [10] Left shin                        (Joint [0])
# [11] Left tarsus                      (Joint [1])
# [12] Left foot             (Motor [4], Joint [2])
# [13] Right hip roll        (Motor [5])
# [14] Right hip yaw         (Motor [6])
# [15] Right hip pitch       (Motor [7])
# [16] Right knee            (Motor [8])
# [17] Right shin                       (Joint [3])
# [18] Right tarsus                     (Joint [4])
# [19] Right foot            (Motor [9], Joint [5])
