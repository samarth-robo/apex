import numpy as np
from . import mujocosim
import time
import torch
from rl import nns
from ..rewards.iros_paper_reward import iros_paper_reward
from ..trajectory.trajectory import CassieTrajectory


class CassieEnv:
    def __init__(self, model_path="../models/cassie.xml", reward_type='standing'):
        self.sim = mujocosim.MujocoSim(model_path)
        self.sim_timestep, self.simrate = 0.0005, 50
        self.obs_dim, self.act_dim = self.cassie_state_fcn(np.zeros(67)).size, 10
        self.reward_type = reward_type

        self.qpos_init = [0, 0, 1.01, 1, 0, 0, 0,
        0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
        -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        self.p_gain = np.array([100,  100,  88,  96,  50,    100,  100,  88,  96,  50])
        self.d_gain = np.array([10.0, 10.0, 8.0, 9.6, 5.0,   10.0, 10.0, 8.0, 9.6, 5.0])

    def step(self, action):
        # motor_pos = self.cassie_state[7:17]
        # motor_vel = self.cassie_state[17:27]
        # target = action + self.offset

        # action = self.p_gain*(target - motor_pos) + self.d_gain*(0 - motor_vel)

        action_scale = np.array( [4.5, 4.5, 12.2, 12.2, 0.9, 4.5, 4.5, 12.2, 12.2, 0.9] )
        action = action_scale * action

        for _ in range(self.simrate):
            state = self.sim.step(action)
        self.cassie_state = self.cassie_state_fcn(state)

        height, vel = state[2], state[35]
        
        if self.reward_type == 'standing':
            motor_pos = self.cassie_state[7:17]
            unactuated_joint_pos = self.cassie_state[27:31]
            left_joint_pos = np.hstack((motor_pos[:5], unactuated_joint_pos[:2]))
            right_joint_pos = np.hstack((motor_pos[5:], unactuated_joint_pos[2:]))
            joint_error = np.mean( (left_joint_pos-right_joint_pos)**2 )
            reward = 1 - joint_error
        elif self.reward_type == 'forward':
            reward = 1 + vel
        else:
            print('No reward type ' + self.reward_type)

        done = not (height > 0.7 and height < 2.0 )

        return self.cassie_state, reward, done

    def reset(self):
        state = self.sim.reset(self.qpos_init)
        self.cassie_state = self.cassie_state_fcn(state)
        
        return self.cassie_state

    def render(self):
        if not hasattr(self, "viewer"):
            self.viewer = mujocosim.MujocoVis(self.sim)
        self.viewer.render()

    def cassie_state_fcn(self,state):
        q_pos, q_vel = state[:35], state[35:]
        pelvis_orientation = q_pos[3:7]         # size 4
        pelvis_angularVel = q_vel[3:6]          # size 3
        motor_pos = [q_pos[7], q_pos[8], q_pos[9], q_pos[14], q_pos[20],   q_pos[21], q_pos[22], q_pos[23], q_pos[28], q_pos[34] ]  # size 10
        motor_vel = [q_vel[6], q_vel[7], q_vel[8], q_vel[12], q_vel[18],   q_vel[19], q_vel[20], q_vel[21], q_vel[25], q_vel[31] ]  # size 10
        joint_pos = [q_pos[15], q_pos[16], q_pos[29], q_pos[30] ]   # size 4
        joint_vel = [q_pos[13], q_pos[14], q_pos[26], q_pos[27] ]   # size 4

        cassie_state = np.hstack((pelvis_orientation, pelvis_angularVel, motor_pos, motor_vel, joint_pos, joint_vel, q_pos[2]))

        return cassie_state
    
    def eval(self, path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']
        actor = nns.MLPGaussianActor(self.obs_dim, self.act_dim, args.hid, torch.nn.ReLU, 0.3)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()
        
        for _ in range(10):
            obs = self.reset()
            for _ in range(500):
                self.render()
                act, _ = actor.action(obs)
                sim_starttime = time.time()
                obs, _, done = self.step(act)
                while time.time() - sim_starttime < self.sim_timestep*self.simrate / play_speed:
                    time.sleep(self.sim_timestep/play_speed)
                if done:
                    break


class CassieWalkingEnv:
    def __init__(self, model_path="../models/cassie.xml", simrate=None, trajdata_path=''):
        self.sim = mujocosim.MujocoSim(model_path)
        self.sim_timestep, self.sim_timestep_default, self.simrate = 0.002, 0.0005, simrate

        self.reward_fn = iros_paper_reward
        self.traj = CassieTrajectory(filepath=trajdata_path)
        self.phase, self.phaseadd = 0, int(self.sim_timestep/self.sim_timestep_default)
        self.phaselen = int(len(self.traj) / self.simrate)

        self.obs_dim, self.act_dim = self.cassie_state_fcn(np.zeros(67)).size, 10

        self.qpos_init = [0, 0, 1.01, 1, 0, 0, 0,
        0.0045, 0, 0.4973, 0.9785, -0.0164, 0.01787, -0.2049,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968,
        -0.0045, 0, 0.4973, 0.9786, 0.00386, -0.01524, -0.2051,
        -1.1997, 0, 1.4267, 0, -1.5244, 1.5244, -1.5968]
        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])
        # self.p_gain = np.array([100,  100,  88,  96,  50,    100,  100,  88,  96,  50])
        # self.d_gain = np.array([10.0, 10.0, 8.0, 9.6, 5.0,   10.0, 10.0, 8.0, 9.6, 5.0])
        self.p_gain = np.array([20.,  20.,  20.,  20.,  1.,    20.,  20.,  20.,  20.,  1.])
        self.d_gain = np.array([2.0, 2.0, 2.0, 2.0, 0.1,   2.0, 2.0, 2.0, 2.0, 0.1])

        # self.action_scale = np.array( [4.5, 4.5, 12.2, 12.2, 0.9, 4.5, 4.5, 12.2, 12.2, 0.9] )
        self.action_scale = np.array([22.5,22.5,70,80,60, 22.5,22.5,70,80,60]) * np.pi/180

        self.motor_pos_idx = [7,8,9,14,20, 21,22,23,28,34]
        self.motor_vel_idx = [6,7,8,12,18, 19,20,21,25,31]

    def step(self, action):
        action = self.action_scale * action
        target = action + self.offset

        for _ in range(self.simrate):
            action = self.step_pd_action(target)
            self.state = self.sim.step(action)
        self.phase += self.phaseadd
        self.cassie_state = self.cassie_state_fcn(self.state)
        
        reward = self.reward_fn(self,self.state)

        height, vel = self.state[2], self.state[35]
        done = (height<0.4) or (height>3.0) or (reward<0.3)

        return self.cassie_state, reward, done

    def step_pd_action(self, target):
        qpos, qvel = self.state[:35], self.state[35:]
        motor_pos = qpos[self.motor_pos_idx]
        motor_vel = qvel[self.motor_vel_idx]
        action = self.p_gain*(target - motor_pos) + self.d_gain*(0 - motor_vel)
        return action

    def reset(self, phase=None):
        self.phase = np.random.randint(0,self.phaselen) if phase is None else phase
        qpos_ini = self.traj.qpos[self.phase*self.simrate]
        qvel_ini = self.traj.qvel[self.phase*self.simrate]
        
        self.state = self.sim.reset(qpos_ini)
        self.cassie_state = self.cassie_state_fcn(self.state)
        
        return self.cassie_state

    def render(self):
        if not hasattr(self, "viewer"):
            self.viewer = mujocosim.MujocoVis(self.sim)
        self.viewer.render()

    def cassie_state_fcn(self,state):
        q_pos, q_vel = state[:35], state[35:]
        pelvis_orientation = q_pos[3:7]         # size 4
        pelvis_angularVel = q_vel[3:6]          # size 3
        motor_pos = [q_pos[7], q_pos[8], q_pos[9], q_pos[14], q_pos[20],   q_pos[21], q_pos[22], q_pos[23], q_pos[28], q_pos[34] ]  # size 10
        motor_vel = [q_vel[6], q_vel[7], q_vel[8], q_vel[12], q_vel[18],   q_vel[19], q_vel[20], q_vel[21], q_vel[25], q_vel[31] ]  # size 10
        joint_pos = [q_pos[15], q_pos[16], q_pos[29], q_pos[30] ]   # size 4
        joint_vel = [q_pos[13], q_pos[14], q_pos[26], q_pos[27] ]   # size 4

        clock = [np.sin(2 * np.pi * self.phase / self.phaselen),
                    np.cos(2 * np.pi * self.phase / self.phaselen)]

        ext_state = np.hstack((q_pos[2],q_vel[2],clock))
        
        cassie_state = np.hstack((pelvis_orientation, pelvis_angularVel, motor_pos, motor_vel, joint_pos, joint_vel, ext_state))

        return cassie_state
    
    def eval(self, path, play_speed=1):
        checkpoint = torch.load(path)
        args = checkpoint['args']
        actor = nns.MLPGaussianActor(self.obs_dim, self.act_dim, args.hid, torch.nn.ReLU, 0.3)
        actor.load_state_dict(checkpoint['actor_state_dict'])
        actor.eval()

        sim_timestep, simrate = self.sim_timestep, self.simrate
        render_rate = int( 0.03 / (sim_timestep*simrate) )
        render_rate = np.max((render_rate,1))
        
        for _ in range(10):
            obs = self.reset()
            for j in range(500):
                sim_starttime = time.time()
                act, _ = actor.action(obs)
                obs, _, done = self.step(act)
                if (j%render_rate) == 0:
                    self.render()
                    while time.time() - sim_starttime < sim_timestep*simrate / play_speed:
                        time.sleep(sim_timestep/play_speed)
                if done:
                    break
        
        # # ------- replay traj data -------
        # for _ in range(1):
        #     phase = 0
        #     simrate = 100
        #     phaselen = int(len(self.traj) / simrate)
        #     for _ in range(50):
        #         sim_starttime = time.time()
        #         ref_pos = self.traj.qpos[phase*simrate]
        #         self.sim.reset(ref_pos)
        #         self.render()
        #         # print('uses: ', time.time() - sim_starttime)    # render time is ~30ms
        #         while time.time() - sim_starttime < self.sim_timestep*simrate:
        #             time.sleep(self.sim_timestep)
        #         phase += 1
        #         if phase >= phaselen:
        #             phase = 0
