```bash
# train Cassie to walk
python apex.py --env Cassie-v001 --simrate 50 --sample_size 10000 --num_procs 10

# evaluate policy
python apex.py --eval log/PPO/Cassie-v001-2020-11-xxxx/Cassie-v001-Epoch1999-proc0.tar
```