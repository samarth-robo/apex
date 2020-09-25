#!/usr/bin/env bash
export MUJOCO_KEY_PATH=/export/share/MujocoLic/Licenses/mujoco/$HOSTNAME/mjkey.txt
python apex.py ppo --env_name CassieKeyframe-v0 --run_name keyframe+stability --reward iros_paper_keyframes --command traj --traj jumping --num_procs 50 --ray_temp_dir /tmp/ray777
