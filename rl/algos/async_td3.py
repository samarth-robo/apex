from rl.utils import ReplayBuffer_remote
from rl.utils import AdaptiveParamNoiseSpec, distance_metric, perturb_actor_parameters
from rl.policies.actor import Scaled_FF_Actor as O_Actor
from rl.policies.critic import Dual_Q_Critic as Critic

from apex import print_logo

import time
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

import gym

import ray

device = torch.device('cpu')



def run_experiment(args):
    torch.set_num_threads(1);
    from apex import env_factory, create_logger

    # Start ray
    ray.init(num_gpus=0, include_webui=True, redis_address=args.redis_address)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # wrapper function for creating parallelized envs
    env_fn = env_factory(args.env_name, state_est=args.state_est, mirror=args.mirror)
    max_traj_len = args.max_traj_len

    obs_dim = env_fn().observation_space.shape[0]
    action_dim = env_fn().action_space.shape[0]
    max_action = 1.0

    # Create remote learner (learner will create the evaluators) and replay buffer
    memory_id = ReplayBuffer_remote.remote(args.replay_size, args.name, args)
    learner_id = Learner.remote(env_fn, memory_id, args.max_timesteps, obs_dim, action_dim, args.a_lr, args.c_lr, batch_size=args.batch_size, discount=args.discount, update_freq=args.update_freq, evaluate_freq=args.evaluate_freq, render_policy=args.render_policy, hidden_size=args.hidden_size, env_name=args.env_name)

    # Create remote actors
    actors_ids = [Actor.remote(env_fn, learner_id, memory_id, action_dim, args.start_timesteps // args.num_procs, args.initial_load_freq, args.taper_load_freq, args.act_noise, args.noise_scale, args.param_noise, i, hidden_size=args.hidden_size, viz_actor=args.viz_actors, env_name=args.env_name) for i in range(args.num_procs)]

    print()
    print("Asynchronous Twin-Delayed Deep Deterministic policy gradients:")
    print("\tenv:            {}".format(args.env_name))
    print("\tmax traj len:   {}".format(args.max_traj_len))
    print("\tseed:           {}".format(args.seed))
    print("\tmirror:         {}".format(args.mirror))
    print("\tnum procs:      {}".format(args.num_procs))
    print("\ta_lr:           {}".format(args.a_lr))
    print("\tc_lr:           {}".format(args.c_lr))
    print("\ttau:            {}".format(args.tau))
    print("\tgamma:          {}".format(args.discount))
    print("\tact noise:      {}".format(args.act_noise))
    print("\tparam noise:    {}".format(args.param_noise))
    if(args.param_noise):
        print("\tnoise scale:    {}".format(args.noise_scale))
    print("\tbatch size:     {}".format(args.batch_size))
    
    # print("\tpolicy noise:   {}".format(args.policy_noise))
    # print("\tnoise clip:     {}".format(args.noise_clip))
    # print("\tpolicy freq:    {}".format(args.policy_freq))

    print("\tload freq:      {}".format(args.initial_load_freq))
    print("\ttaper load freq:{}".format(args.taper_load_freq))
    print()

    start = time.time()

    # start collection loop for each actor
    futures = [actor_id.collect_experience.remote() for actor_id in actors_ids]

    # start training loop for learner
    # while True:
    #     learner_id.update_model.remote()

    # TODO: make evaluator its own ray object with separate loop
    # futures.append(evaluator_id...)

    # wait for training to complete (THIS DOESN'T WORK AND I DON'T KNOW WHY)
    ray.wait(futures, num_returns=len(futures))


def select_action(perturbed_policy, unperturbed_policy, state, device, param_noise=None):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)

    unperturbed_policy.eval()

    if param_noise is not None:
        return perturbed_policy(state).cpu().data.numpy().flatten()
    else:
        return unperturbed_policy(state).cpu().data.numpy().flatten()

def select_greedy_action(Policy, state, device):
    state = torch.FloatTensor(state.reshape(1, -1)).to(device)

    Policy.eval()

    return Policy(state).cpu().data.numpy().flatten()

@ray.remote
def evaluator(env, policy, max_traj_len, render_policy=False):

    env = env()

    state = env.reset()
    total_reward = 0
    total_steps = 0
    steps = 0
    done = False

    # evaluate performance of the passed model for one episode
    while steps < max_traj_len and not done:
        if render_policy:
            env.render()

        # use model's greedy policy to predict action
        action = select_greedy_action(policy, np.array(state), device)

        # take a step in the simulation
        next_state, reward, done, _ = env.step(action)

        # update state
        state = next_state

        # increment total_reward and step count
        total_reward += reward
        steps += 1
        total_steps += 1

    return total_reward, total_steps

class ActorBuffer(object):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self.storage = []
        self.max_size = size
        self.ptr = 0

    def __len__(self):
        return len(self.storage)

    def storage_size(self):
        return len(self.storage)

    def add(self, data):
        if len(self.storage) < self.max_size:
            self.storage.append(data)
        self.storage[int(self.ptr)] = data
        self.ptr = (self.ptr + 1) % self.max_size
        #print("Added experience to replay buffer.")

    def get_transitions(self):
        ind = np.arange(0, len(self.storage))
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        #print("Sampled experience from replay buffer.")

        self.clear()

        return np.array(x), np.array(u)

    def clear(self):
        self.storage = []
        self.ptr = 0

    def send_to_global_replay(self, remote_replay_id):
        ray.wait([remote_replay_id.add_bulk.remote(self.storage)])
        # print("send experience over")
        self.clear()


@ray.remote
class Actor():
    def __init__(self, env_fn, learner_id, memory_id, action_dim, start_timesteps, load_freq, taper_load_freq, act_noise, noise_scale, param_noise, id, hidden_size=256, viz_actor=True, env_name='NOT_SET'):

        self.device = torch.device('cpu')

        self.env = env_fn()
        self.cassieEnv = True

        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        #self.max_action = float(self.env.action_space.high[0])
        self.max_action = 1

        #self.policy = LN_Actor(self.state_dim, self.action_dim, self.max_action, 400, 300).to(device)
        self.learner_id = learner_id
        self.memory_id = memory_id

        # Action noise
        self.start_timesteps = start_timesteps
        self.act_noise = act_noise

        # Initialize param noise (or set to none)
        self.noise_scale = noise_scale
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.05, desired_action_stddev=self.noise_scale, adaptation_coefficient=1.05) if param_noise else None
        self.policy_perturbed = O_Actor(self.state_dim, self.action_dim, self.max_action, hidden_size=hidden_size, env_name=env_name).to(self.device)

        # Termination condition: max episode length
        self.max_traj_len = 400

        # Counters
        self.actor_timesteps = 0
        self.taper_timesteps = 0
        self.episode_num = 0
        self.taper_load_freq = taper_load_freq  # taper load freq or not?
        # initial load frequency... make this taper down to 1 over time
        self.load_freq = load_freq

        # Local replay buffer
        self.local_buffer = ActorBuffer(self.max_traj_len * self.load_freq)

        self.id = id

        self.viz_actor = viz_actor

        self.policy, self.training_done = ray.get(self.learner_id.get_global_policy.remote())

    def collect_experience(self):

        print("Actor {} starting collection".format(self.id))

        while not self.training_done:

            if self.actor_timesteps % self.load_freq == 0:
                # PUTTING WAIT ON THIS SHOULD MAKE THIS EXACT SAME AS NON-DISTRIBUTED, if using one actor
                # Query learner for latest model and termination flag

                # time this for debugging (need to implement sharded parameter server or not)
                # start = time.time()

                self.policy, self.training_done = ray.get(self.learner_id.get_global_policy.remote())

                # duration = time.time() - start

                #global_policy_state_dict, training_done = ray.get(self.learner_id.get_global_policy.remote())
                # self.policy.load_state_dict(global_policy_state_dict)

                # If we have loaded a global model, we also need to update the param_noise based on the distance metric
                if self.param_noise is not None:
                    states, perturbed_actions = self.local_buffer.get_transitions()
                    unperturbed_actions = np.array([select_action(
                        self.policy_perturbed, self.policy, state, device, param_noise=None) for state in states])
                    dist = distance_metric(perturbed_actions, unperturbed_actions)
                    self.param_noise.adapt(dist)
                    # print("loaded global model and adapted parameter noise. Load duration = {}".format(duration))
                    #print("loaded global model and adapted parameter noise")
                else:
                    # print("loaded global model.  Load duration = {}".format(duration))
                    # print("loaded global model.")
                    pass

            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_timesteps = 0

            # nested collection loop - COLLECTS TIMESTEPS OF EXPERIENCE UNTIL episode is over
            while episode_timesteps < self.max_traj_len and not done:

                # self.env.render()

                # Param Noise
                if self.param_noise:
                    perturb_actor_parameters(self.policy_perturbed, self.policy, self.param_noise, device)

                # Select action randomly or according to policy
                if self.actor_timesteps < self.start_timesteps:
                    #print("selecting action randomly {}".format(done_bool))
                    action = torch.randn(
                        self.env.action_space.shape[0]) if self.cassieEnv is True else self.env.action_space.sample()
                    action = action.numpy()
                else:
                    #print("selecting from policy")
                    action = select_action(self.policy_perturbed, self.policy, np.array(obs), device, param_noise=self.param_noise)
                    if self.act_noise != 0:
                        # action = (action + np.random.normal(0, self.act_noise,
                        #                                     size=self.env.action_space.shape[0])).clip(self.env.action_space.low, self.env.action_space.high)
                        action = (action + np.random.normal(0, self.act_noise,
                                                                size=self.env.action_space.shape[0])).clip(-1, 1)

                # Perform action
                new_obs, reward, done, _ = self.env.step(action)
                done_bool = 1.0 if episode_timesteps + \
                    1 == self.max_traj_len else float(done)
                episode_reward += reward

                # Store data in local replay buffer
                transition = (obs, new_obs, action, reward, done_bool)
                self.local_buffer.add(transition)
                # self.memory_id.add.remote(transition)

                # # call update from model server
                # self.learner_id.update_model.remote(iterations=1)
                #= self.learner_id.update_eval_model.remote()

                # update state
                obs = new_obs

                # increment step counts
                episode_timesteps += 1
                self.actor_timesteps += 1

                # TODO: Is this inefficient because of how many actors there are?
                # increment global step count
                self.learner_id.increment_step_count.remote()

            # episode is over, increment episode count and plot episode info
            self.episode_num += 1

            # dump transitions from local buffer into global replay buffer (blocking call)
            self.local_buffer.send_to_global_replay(self.memory_id)

            # tell learner to update
            self.learner_id.update_model.remote()

            # pass episode details to visdom logger on memory server
            if(self.viz_actor):
                self.memory_id.plot_actor_results.remote(self.id, self.actor_timesteps, episode_reward)

            # # TODO: check if this is inefficient
            # # increment episode count and wait for that to complete
            # ray.wait([self.learner_id.increment_episode_count.remote()], num_returns=1)

            if self.taper_load_freq and self.taper_timesteps >= 2000:
                self.load_freq = self.load_freq // 2
                print("Increased load frequency")

                


@ray.remote(num_gpus=0)
class Learner():
    def __init__(self, env_fn, memory_server, max_timesteps, state_space, action_space, a_lr, c_lr,
                 batch_size=500, discount=0.99, tau=0.005, update_freq=10,
                 target_update_freq=2000, evaluate_freq=1000, render_policy=True, hidden_size=256, env_name='NOT_SET'):

        self.device = torch.device('cpu')

        # keep uninstantiated constructor for evaluator
        self.env_fn = env_fn

        self.env = env_fn()
        self.max_timesteps = max_timesteps

        self.batch_size = batch_size

        self.update_freq = update_freq
        self.target_update_freq = target_update_freq
        self.evaluate_freq = evaluate_freq     # how many steps before each eval

        self.num_of_evaluators = 4

        # counters
        self.step_count = 0                     # global step count
        self.eval_step_count = 0
        self.target_step_count = 0

        self.update_counter = 0

        # hyperparams
        self.discount = discount
        self.tau = tau

        # results list
        self.results = []

        # highest reward tracker
        self.highest_return = 0

        # experience replay
        self.memory = memory_server

        # env attributes
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        #self.max_action = float(self.env.action_space.high[0])
        self.max_action = 1
        self.max_traj_len = 400

        # models and optimizers
        self.actor = O_Actor(self.state_dim, self.action_dim, self.max_action, hidden_size=hidden_size, env_name=env_name).to(self.device)
        self.actor_target = O_Actor(self.state_dim, self.action_dim, self.max_action, hidden_size=hidden_size, env_name=env_name).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)

        self.critic = Critic(self.state_dim, self.action_dim, hidden_size=hidden_size, env_name=env_name).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, hidden_size=hidden_size, env_name=env_name).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=c_lr)

        # render policy? This doesn't do anything atm
        self.render_policy = render_policy

        # start time for logging duration later
        self.start_time = time.time()

    def train(self):
        while self.step_count < self.max_timesteps:
            self.update_model()

    def increment_step_count(self):
        self.step_count += 1        # global step count
        self.eval_step_count += 1   # eval step count

    # def increment_episode_count(self):
    #     self.episode_count += 1
    #     self.eval_episode_count += 1
    #     # print(self.episode_count)

    def is_training_finished(self):
        return self.step_count >= self.max_timesteps

    def update_model(self, policy_noise=0.2, noise_clip=0.5, policy_freq=2):

        foo = ray.get(self.memory.storage_size.remote())

        if foo < self.batch_size:
            # print("not enough experience yet: {}".format(foo))
            return

        start_time = time.time()

        # randomly sample a mini-batch transition from memory_server
        x, y, u, r, d = ray.get(self.memory.sample.remote(self.batch_size))
        #print("sampled from replay buffer. Duration = {}".format(time.time() - start_time))
        state = torch.FloatTensor(x).to(self.device)
        action = torch.FloatTensor(u).to(self.device)
        next_state = torch.FloatTensor(y).to(self.device)
        done = torch.FloatTensor(1 - d).to(self.device)
        reward = torch.FloatTensor(r).to(self.device)

        # Select action according to policy and add clipped noise
        noise = torch.FloatTensor(u).data.normal_(
            0, policy_noise).to(self.device)
        noise = noise.clamp(-noise_clip, noise_clip)
        next_action = (self.actor_target(next_state) +
                        noise).clamp(-self.max_action, self.max_action)

        # Compute the target Q value
        target_Q1, target_Q2 = self.critic_target(next_state, next_action)
        target_Q = torch.min(target_Q1, target_Q2)
        target_Q = reward + (done * self.discount * target_Q).detach()

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(
            current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.memory.plot_critic_loss.remote(self.update_counter, critic_loss, torch.mean(current_Q1), torch.mean(current_Q2))

        self.update_counter += 1            

        # Delayed policy updates
        if self.update_counter % policy_freq == 0:

            # print("optimizing at timestep {} | time = {} | replay size = {} | update count = {} ".format(self.step_count, time.time()-start_time, ray.get(self.memory.storage_size.remote()), self.update_counter))

            # Compute actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data)

            self.memory.plot_actor_loss.remote(self.update_counter, actor_loss)

            # Evaluate and possibly save
            if self.eval_step_count > self.evaluate_freq:
                self.eval_step_count = 0
                self.results.append(self.evaluate(trials=self.num_of_evaluators))

                # save policy if it has highest return so far
                if self.highest_return < self.results[-1]:
                    self.highest_return = self.results[-1]
                    self.save()
            
        #print("optimize time elapsed: {}".format(time.time() - start_time))
        

    # TODO: make evaluator another remote actor to speed this up (currently bottleneck)
    def evaluate(self, trials=30, render_policy=True):

        print("starting evaluation")

        start_time = time.time()

        # initialize evaluators
        evaluators = [evaluator.remote(self.env_fn, self.actor, max_traj_len=400)
                      for _ in range(self.num_of_evaluators)]

        total_rewards = 0
        total_eplen = 0

        for t in range(trials):
            # get result from a worker
            ready_ids, _ = ray.wait(evaluators, num_returns=1)

            # update total rewards
            rewards, eplens = ray.get(ready_ids[0])
            total_rewards += rewards
            total_eplen += eplens

            # remove ready_ids from the evaluators
            evaluators.remove(ready_ids[0])

            # start a new worker
            evaluators.append(evaluator.remote(self.env_fn, self.actor, self.max_traj_len))

        # return average reward
        avg_reward = total_rewards / trials
        avg_eplen = total_eplen / trials
        self.memory.plot_eval_results.remote(self.step_count, avg_reward, avg_eplen, self.update_counter)

        # tell replay to plot hist of actor policy weights
        self.memory.plot_policy_hist.remote(self.actor, self.update_counter)

        return avg_reward

    def test(self):
        return 0

    def get_global_policy(self):
        #print("returning global policy")
        return self.actor, self.is_training_finished()

    def get_global_timesteps(self):
        return self.step_count

    def get_results(self):
        return self.results, self.evaluate_freq

    def save(self):
        if not os.path.exists('trained_models/asyncTD3/'):
            os.makedirs('trained_models/asyncTD3/')

        print("Saving model")

        filetype = ".pt"  # pytorch model
        torch.save(self.actor.state_dict(), os.path.join(
            "./trained_models/asyncTD3", "actor_model" + filetype))
        torch.save(self.critic.state_dict(), os.path.join(
            "./trained_models/asyncTD3", "critic_model" + filetype))

    def load(self, model_path):
        actor_path = os.path.join(model_path, "actor_model.pt")
        critic_path = os.path.join(model_path, "critic_model.pt")
        print('Loading models from {} and {}'.format(actor_path, critic_path))
        if actor_path is not None:
            self.actor.load_state_dict(torch.load(actor_path))
            self.actor.eval()
        if critic_path is not None:
            self.critic.load_state_dict(torch.load(critic_path))
            self.critic.eval()
