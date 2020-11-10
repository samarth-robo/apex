import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from . import nns
from . import mpi_tools
from cassie import env_tools
import time, sys



class PPOBuffer:
    def __init__(self, obs_dim, act_dim, buffer_size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((buffer_size,obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((buffer_size,act_dim), dtype=np.float32)
        self.reward_buf = np.zeros(buffer_size, dtype=np.float32)
        self.reward_to_go_buf = np.zeros(buffer_size, dtype=np.float32)
        self.logp_buff = np.zeros(buffer_size, dtype=np.float32)
        
        self.gamma = gamma
        self.ptr, self.traj_start_idx = 0, 0

    def store(self, obs, act, reward, logp):
        self.obs_buf[self.ptr,:] = obs
        self.act_buf[self.ptr,:] = act
        self.reward_buf[self.ptr] = reward
        self.logp_buff[self.ptr] = logp
        
        self.ptr += 1
    
    def finish_traj(self, last_val=0):
        rewards = self.reward_buf[self.traj_start_idx:self.ptr]
        r_cum = np.zeros_like(rewards)

        r_cum[-1] = rewards[-1] + last_val
        for k in reversed(range(len(rewards)-1)):
            r_cum[k] = rewards[k] + self.gamma*r_cum[k+1]

        self.reward_to_go_buf[self.traj_start_idx:self.ptr] = r_cum

        self.traj_start_idx = self.ptr

    def get(self):
        self.ptr, self.traj_start_idx = 0, 0
        return self.obs_buf, self.act_buf, self.reward_buf, self.reward_to_go_buf, self.logp_buff


class rlPPO:
    def __init__(self, actor:nns.MLPGaussianActor, critic:nns.MLPCritic, lr_a, lr_c, train_a_itrs, train_c_itrs, log_dir_name):
        self.actor, self.critic= actor, critic

        self.optimizer_a = torch.optim.Adam(self.actor.parameters(), lr=lr_a)
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)
        self.train_a_itrs, self.train_c_itrs = train_a_itrs, train_c_itrs

        self.proc_id = mpi_tools.proc_id()
        self.log_dir = './log/PPO/' + log_dir_name + time.strftime( '-%Y-%m-%d_%H-%M-%S', time.localtime(time.time()) )
        if self.proc_id == 0:
            self.writer = SummaryWriter(log_dir = self.log_dir)

    def action(self, x):
        return self.actor.action(x)

    def train(self, buffer:PPOBuffer):
        obs, acts, rewards, r2g, logp_old = buffer.get()

        obs = torch.tensor(obs, dtype=torch.float32)
        acts = torch.tensor(acts, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        r2g = torch.tensor(r2g, dtype=torch.float32)
        logp_old = torch.tensor(logp_old, dtype=torch.float32)

        baselines = self.critic(obs).detach().view(-1)
        advantages = r2g - baselines

        for _ in range(self.train_a_itrs):
            pi = self.actor(obs)
            logp = pi.log_prob(acts).sum(axis=-1)
            ratios = (logp - logp_old).exp()
            loss_a = -torch.min( advantages*ratios, advantages*torch.clamp(ratios, 1-0.2, 1+0.2) ).mean()

            self.optimizer_a.zero_grad()
            loss_a.backward()
            mpi_tools.mpi_avg_grads(self.actor)
            self.optimizer_a.step()
        
        for _ in range(self.train_c_itrs):
            self.optimizer_c.zero_grad()
            loss_c = F.mse_loss(self.critic(obs), r2g.view(-1,1))
            loss_c.backward()
            mpi_tools.mpi_avg_grads(self.critic)
            self.optimizer_c.step()
    
    def write_reward(self, k_epoch, reward, ep_len):
        if self.proc_id == 0:
            self.writer.add_scalar("Reward", reward, k_epoch)
            self.writer.add_scalar("Ep_len", ep_len, k_epoch)

    def save_checkpoint(self, filename, args):
        path = self.log_dir + '/' + filename
        torch.save( {'actor_state_dict': self.actor.state_dict(),
                    'critic_state_dict': self.critic.state_dict(),
                    'actor_optimizer_state_dict': self.optimizer_a.state_dict(),
                    'critic_optimizer_state_dict': self.optimizer_c.state_dict(),
                    'args': args},
                    path )


def run_experiments(args):
    if args.eval:
        eval_path = args.eval
        play_speed = args.play_speed
        args = torch.load(eval_path)['args']
        print('Evaluate ' + args.env)
        env = env_tools.env_by_name(args)
        env.eval(path=eval_path, play_speed=play_speed)
        sys.exit()

    mpi_tools.mpi_fork(args.num_procs)
    proc_id = mpi_tools.proc_id()

    mpi_tools.setup_pytorch_for_mpi()

    # Random seed
    seed = 0
    seed += 10000 * proc_id
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    env = env_tools.env_by_name(args)
    sample_size = int( args.sample_size / mpi_tools.num_procs() )
    print('sample_size ', sample_size, flush=True)

    buffer = PPOBuffer(obs_dim=env.obs_dim, act_dim=env.act_dim, buffer_size=sample_size, gamma=args.gamma)
    actor_std = 0.3*np.ones(env.act_dim, dtype=np.float32)
    actor = nns.MLPGaussianActor(env.obs_dim, env.act_dim, args.hid, torch.nn.ReLU, actor_std)
    critic = nns.MLPCritic(env.obs_dim, args.hid, torch.nn.ReLU)
    mpi_tools.sync_params(actor)
    mpi_tools.sync_params(critic)
    agent = rlPPO(actor, critic, lr_a=args.lr_a, lr_c=args.lr_c, train_a_itrs=args.train_a_itrs, train_c_itrs=args.train_c_itrs, log_dir_name=args.env)

    for k_epoch in range(args.epochs):
        obs = env.reset()
        epoch_start_time = time.time()

        k_ep = 0
        k_sample = 0
        while True:
            k_sample += 1
            # env.render()
            
            act, logp = agent.action(obs)
            obs_, reward, done = env.step(act)

            buffer.store(obs, act, reward, logp)

            obs = obs_

            if k_sample==sample_size:
                k_ep += 1
                if done:
                    last_val=0
                else:
                    last_val = agent.critic(torch.tensor(obs,dtype=torch.float32)).detach().numpy()
                buffer.finish_traj(last_val=last_val)
                break
            
            if done:
                k_ep += 1
                buffer.finish_traj()
                env.reset()
        
        sampling_time = time.time() - epoch_start_time
        agent.train(buffer)
        training_time = time.time() - epoch_start_time - sampling_time

        if proc_id == 0:
            print( 'Epoch: %3d \t return: %.3f \t ep_len: %.3f' %(k_epoch, np.mean(buffer.reward_to_go_buf), sample_size/k_ep), flush=True )
            print('Tsp: %.3f \t Ttr: %.3f' %(sampling_time, training_time), flush=True)
            agent.write_reward(k_epoch, np.mean(buffer.reward_to_go_buf), sample_size/k_ep)
            if k_epoch % args.save_freq == args.save_freq-1:
                agent.save_checkpoint(filename=args.env+'-Epoch'+str(k_epoch)+'-proc' + str(proc_id)+'.tar', args=args)
        