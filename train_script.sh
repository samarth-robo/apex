#!/usr/bin/env bash
export MUJOCO_KEY_PATH=/export/share/MujocoLic/Licenses/mujoco/$HOSTNAME/mjkey.txt
python apex.py ppo --env_name CassieTraj-v0 --run_name interp_ts0.4_v2 --reward iros_paper --command traj --traj jumping --num_procs 80 --ray_temp_dir /tmp/ray777 --bounded --n_bounded 1
