import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import time
import math
import random
import ray
from functools import partial

# Quaternion utility functions. Due to python relative imports and directory structure can't cleanly use cassie.quaternion_function
def inverse_quaternion(quaternion):
	result = np.copy(quaternion)
	result[1:4] = -result[1:4]
	return result

def quaternion_product(q1, q2):
	result = np.zeros(4)
	result[0] = q1[0]*q2[0]-q1[1]*q2[1]-q1[2]*q2[2]-q1[3]*q2[3]
	result[1] = q1[0]*q2[1]+q2[0]*q1[1]+q1[2]*q2[3]-q1[3]*q2[2]
	result[2] = q1[0]*q2[2]-q1[1]*q2[3]+q1[2]*q2[0]+q1[3]*q2[1]
	result[3] = q1[0]*q2[3]+q1[1]*q2[2]-q1[2]*q2[1]+q1[3]*q2[0]
	return result

def rotate_by_quaternion(vector, quaternion):
	q1 = np.copy(quaternion)
	q2 = np.zeros(4)
	q2[1:4] = np.copy(vector)
	q3 = inverse_quaternion(quaternion)
	q = quaternion_product(q2, q3)
	q = quaternion_product(q1, q)
	result = q[1:4]
	return result

def euler2quat(z=0, y=0, x=0):
    z = z/2.0
    y = y/2.0
    x = x/2.0
    cz = math.cos(z)
    sz = math.sin(z)
    cy = math.cos(y)
    sy = math.sin(y)
    cx = math.cos(x)
    sx = math.sin(x)
    result =  np.array([
             cx*cy*cz - sx*sy*sz,
             cx*sy*sz + cy*cz*sx,
             cx*cz*sy - sx*cy*sz,
             cx*cy*sz + sx*cz*sy])
    if result[0] < 0:
    	result = -result
    return result

@ray.remote
@torch.no_grad()
def eval_commands_worker(env_fn, policy, num_steps, num_commands, max_speed, min_speed, num_iters):
    cassie_env = env_fn()
    # save_data will hold whether passed or not (1 or 0), whether orient command or speed command caused failure (1, 0),
    # speed and orient command at failure, and speed and orient change at failure
    save_data = np.zeros((num_iters, 6))
    start_t = time.time()
    for j in range(num_iters):
        state = torch.Tensor(cassie_env.reset_for_test())
        cassie_env.speed = 0.5
        cassie_env.side_speed = 0
        cassie_env.phase_add = 1
        speed_schedule = [0.5]
        for i in range(num_commands-1):
            speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
            if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
                speed_add *= -1
            speed_schedule.append(speed_schedule[i] + speed_add)
        orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
        orient_sign = np.random.choice((-1, 1), num_commands)
        orient_schedule = orient_schedule * orient_sign
        count = 0
        orient_ind = 0
        speed_ind = 1 
        orient_add = 0
        passed = 1
        while not (speed_ind == num_commands and orient_ind == num_commands and count == num_steps) and passed:
            if count == num_steps:
                count = 0
                cassie_env.speed = speed_schedule[speed_ind]
                cassie_env.speed = np.clip(cassie_env.speed, min_speed, max_speed)
                if cassie_env.speed > 1.4:
                    cassie_env.phase_add = 1.5
                else:
                    cassie_env.phase_add = 1
                speed_ind += 1
            elif count == num_steps // 2:
                orient_add += orient_schedule[orient_ind]
                orient_ind += 1
            # Update orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            curr_orient = state[1:5]
            curr_transvel = state[15:18]

            new_orient = quaternion_product(iquaternion, curr_orient)
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            state[1:5] = torch.FloatTensor(new_orient)
            state[15:18] = torch.FloatTensor(new_translationalVelocity)

            # Get action
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            if cassie_env.sim.qpos()[2] < 0.4:
                passed = 0
            count += 1
        if passed:
            save_data[j, 0] = passed
            save_data[j, 1] = -1
        else:
            save_data[j, :] = np.array([passed, count//(num_steps//2), cassie_env.speed, orient_add,\
                        cassie_env.speed-speed_schedule[max(0, speed_ind-2)], orient_schedule[orient_ind-1]])
            # if save_data[j, 1] == 0:
                # print("speed diff: ", speed_schedule[speed_ind-1]-speed_schedule[speed_ind-2])
                # print("curr speed: ", cassie_env.speed)
                # print("speed schedule: ", speed_schedule)
                # print("speed ind: ", speed_ind)
                # print("curr schedule: ", speed_schedule[speed_ind-1])
    return save_data, time.time() - start_t

def eval_commands_multi(env_fn, policy, num_steps=200, num_commands=4, max_speed=3, min_speed=0, num_iters=4, num_procs=4, filename="test_eval_command.npy"):
    start_t1 = time.time()
    ray.init(num_cpus=num_procs)
    result_ids = []
    for i in range(num_procs):
        curr_iters = num_iters // num_procs
        if i == num_procs - 1:  # is last process to get launched, do remaining iters if not evenly divided between procs
            curr_iters = num_iters - i*curr_iters
        print("curr iters: ", curr_iters)
        args = (env_fn, policy, num_steps, num_commands, max_speed, min_speed, curr_iters)
        print("Starting worker ", i)
        result_ids.append(eval_commands_worker.remote(*args))
    result = ray.get(result_ids)
    # print(result)
    print("Got all results")
    total_data = np.concatenate([result[i][0] for i in range(num_procs)], axis=0)
    # print("timings: ", [result[i][1] for i in range(num_procs)])
    # print("sim timings: ", [result[i][2] for i in range(num_procs)])
    # # max_force = np.concatenate(result, axis=1)
    # print("total_data: ", total_data)
    np.save(filename, total_data)
    print("total time: ", time.time() - start_t1)

def report_stats(filename):
    data = np.load(filename)
    num_iters = data.shape[0]
    pass_rate = np.sum(data[:, 0]) / num_iters
    success_inds = np.where(data[:, 0] == 1)[0]
    # data[success_inds, 1] = -1
    speed_fail_inds = np.where(data[:, 1] == 0)[0]
    orient_fail_inds = np.where(data[:, 1] == 1)[0]
    print("pass rate: ", pass_rate)
    # print("speed failure: ", data[speed_fail_inds, 4])
    # print("orient failure: ", data[orient_fail_inds, 5])
    speed_change = data[speed_fail_inds, 4]
    orient_change = data[orient_fail_inds, 5]
    speed_neg_inds = np.where(speed_change < 0)
    speed_pos_inds = np.where(speed_change > 0)
    orient_neg_inds = np.where(orient_change < 0)
    orient_pos_inds = np.where(orient_change > 0)
    print("Number of speed failures: ", len(speed_fail_inds))
    print("Number of orient failures: ", len(orient_fail_inds))
    if len(speed_fail_inds) == 0:
        avg_pos_speed = "N/A"
        avg_neg_speed = "N/A"
    else:
        avg_pos_speed = np.mean(speed_change[speed_pos_inds])
        avg_neg_speed = np.mean(speed_change[speed_neg_inds])
    if len(orient_fail_inds) == 0:
        avg_pos_orient = "N/A"
        avg_neg_orient = "N/A"
    else:
        avg_pos_orient = np.mean(orient_change[orient_pos_inds])
        avg_neg_orient = np.mean(orient_change[orient_neg_inds])
    print("avg pos speed failure: ", avg_pos_speed)
    print("avg neg speed failure: ", avg_neg_speed)
    print("avg pos orient failure: ", avg_pos_orient)
    print("avg neg orient failure: ", avg_neg_orient)


@torch.no_grad()
def eval_commands(cassie_env, policy, num_steps=200, num_commands=2, max_speed=3, min_speed=0, num_iters=1):
    # save_data will hold whether passed or not (1 or 0), whether orient command or speed command caused failure (1, 0),
    # speed and orient command at failure, and speed and orient change at failure
    save_data = np.zeros((num_iters, 6))
    start_t = time.time()
    for j in range(num_iters):
        state = torch.Tensor(cassie_env.reset_for_test())
        cassie_env.speed = 0.5
        cassie_env.side_speed = 0
        cassie_env.phase_add = 1
        speed_schedule = [0.5]
        for i in range(num_commands-1):
            speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
            if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
                speed_add *= -1
            speed_schedule.append(speed_schedule[i] + speed_add)
        orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
        orient_sign = np.random.choice((-1, 1), num_commands)
        orient_schedule = orient_schedule * orient_sign
        # print("Speed schedule: ", speed_schedule)
        # print("Orient schedule: ", orient_schedule)
        count = 0
        orient_ind = 0
        speed_ind = 1 
        orient_add = 0
        passed = 1
        while not (speed_ind == num_commands and orient_ind == num_commands and count == num_steps) and passed:
            if count == num_steps:
                count = 0
                cassie_env.speed = speed_schedule[speed_ind]
                cassie_env.speed = np.clip(cassie_env.speed, min_speed, max_speed)
                if cassie_env.speed > 1.4:
                    cassie_env.phase_add = 1.5
                else:
                    cassie_env.phase_add = 1
                speed_ind += 1
                # print("Current speed: ", cassie_env.speed, speed_ind)
            elif count == num_steps // 2:
                orient_add += orient_schedule[orient_ind]
                orient_ind += 1
                # print("Current orient add: ", orient_add, orient_ind)
            # Update orientation
            quaternion = euler2quat(z=orient_add, y=0, x=0)
            iquaternion = inverse_quaternion(quaternion)
            curr_orient = state[1:5]
            curr_transvel = state[15:18]

            new_orient = quaternion_product(iquaternion, curr_orient)
            if new_orient[0] < 0:
                new_orient = -new_orient
            new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
            state[1:5] = torch.FloatTensor(new_orient)
            state[15:18] = torch.FloatTensor(new_translationalVelocity)

            # Get action
            action = policy(state, True)
            action = action.data.numpy()
            state, reward, done, _ = cassie_env.step(action)
            state = torch.Tensor(state)
            if cassie_env.sim.qpos()[2] < 0.4:
                # print("Failed")
                passed = 0
            count += 1
        if passed:
            # print("passed")
            save_data[j, 0] = passed
            save_data[j, 1] = -1
        else:
            # print("didnt pass")
            save_data[j, :] = np.array([passed, count//(num_steps//2), cassie_env.speed, orient_add,\
                        cassie_env.speed-speed_schedule[max(0, speed_ind-2)], orient_schedule[orient_ind-1]])
    print("time: ", time.time() - start_t)
    return save_data

def vis_commands(cassie_env, policy, num_steps=200, num_commands=4, max_speed=1, min_speed=0):
    state = torch.Tensor(cassie_env.reset_for_test())

    cassie_env.speed = 0.5
    cassie_env.side_speed = 0
    cassie_env.phase_add = 1
    # orient_schedule = np.pi/4*np.arange(8)
    # speed_schedule = np.random.uniform(-1.5, 1.5, 4)
    speed_schedule = [0.5]
    for i in range(num_commands-1):
        speed_add = random.choice([-1, 1])*random.uniform(0.4, 1.3)
        if speed_schedule[i] + speed_add < min_speed or speed_schedule[i] + speed_add > max_speed:
            speed_add *= -1
        speed_schedule.append(speed_schedule[i] + speed_add)
    orient_schedule = np.random.uniform(np.pi/6, np.pi/3, num_commands)
    orient_sign = np.random.choice((-1, 1), num_commands)
    orient_schedule = orient_schedule * orient_sign
    print("Speed schedule: ", speed_schedule)
    print("Orient schedule: ", orient_schedule)
    dt = 0.05
    speedup = 3
    count = 0
    orient_ind = 0
    speed_ind = 0 
    orient_add = 0
    # print("Current orient add: ", orient_add)

    render_state = cassie_env.render()
    with torch.no_grad():
        while render_state:
            if (not cassie_env.vis.ispaused()):
                # orient_add = orient_schedule[math.floor(count/num_steps)]
                if count == num_steps:
                    count = 0
                    speed_ind += 1
                    if speed_ind >= len(speed_schedule):
                        print("speed Done")
                        exit()
                    cassie_env.speed = speed_schedule[speed_ind]
                    cassie_env.speed = np.clip(cassie_env.speed, 0, 3)
                    if cassie_env.speed > 1.4:
                        cassie_env.phase_add = 1.5
                    print("Current speed: ", cassie_env.speed)
                elif count == num_steps // 2:
                    orient_ind += 1
                    if orient_ind >= len(orient_schedule):
                        print("orient Done")
                        exit()
                    orient_add += orient_schedule[orient_ind]
                    print("Current orient add: ", orient_add)
                # Update orientation
                quaternion = euler2quat(z=orient_add, y=0, x=0)
                iquaternion = inverse_quaternion(quaternion)
                curr_orient = state[1:5]
                curr_transvel = state[15:18]

                new_orient = quaternion_product(iquaternion, curr_orient)
                if new_orient[0] < 0:
                    new_orient = -new_orient
                new_translationalVelocity = rotate_by_quaternion(curr_transvel, iquaternion)
                state[1:5] = torch.FloatTensor(new_orient)
                state[15:18] = torch.FloatTensor(new_translationalVelocity)

                # Get action
                action = policy(state, True)
                action = action.data.numpy()
                state, reward, done, _ = cassie_env.step(action)
                if cassie_env.sim.qpos()[2] < 0.4:
                    print("Failed")
                    exit()
                else:
                    state = torch.Tensor(state)
                count += 1
                
            render_state = cassie_env.render()
            time.sleep(dt / speedup)

# report_stats("./5k_orientadd.npy")
# exit()



# Load environment and policy
# cassie_env = CassieEnv("walking", clock_based=True, state_est=False)
# cassie_env = CassieEnv(traj="walking", clock_based=True, state_est=True, dynamics_randomization=False, no_delta=True)

# obs_dim = cassie_env.observation_space.shape[0] # TODO: could make obs and ac space static properties
# action_dim = cassie_env.action_space.shape[0]

# no_delta = True
# offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

# # file_prefix = "fwrd_walk_StateEst_speed-05-3_freq1-2_footvelpenalty_heightflag_footxypenalty"
# # file_prefix = "sidestep_StateEst_speedmatch_footytraj_doublestance_time0.4_land0.2_vels_avgdiff_simrate15_evenweight_actpenalty"
# file_prefix = "nodelta_neutral_StateEst_symmetry_speed0-3_freq1-2"
# # file_prefix = "5k_speedschedule_orientadd_torquecost_smoothcost"
# policy = torch.load("./trained_models/new_policies/{}_actor.pt".format(file_prefix))
# policy.bounded = False
# policy.eval()

# vis_commands(cassie_env, policy, num_steps=200, num_commands=6, max_speed=3, min_speed=0)
# save_data = eval_commands(cassie_env, policy, 200, 2, 3, 0, 1)
# np.save(file_prefix+"_eval_commands.npy", save_data)
# eval_commands_multi(env_fn, policy, 200, 6, 3, 0, 10000, 100, "./5k_orientadd.npy")
# report_stats("./5k_orientadd.npy")