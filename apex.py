import argparse, sys
from rl.rlPPO import run_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # ------------ Cassie-v0 ------------
    parser.add_argument('--env', type=str, default='Cassie-v001')
    parser.add_argument('--simrate', type=int, default=50)
    parser.add_argument('--sample_size', type=int, default=5000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--hid', type=list, default=[256,256])
    parser.add_argument('--lr_a', type=float, default=5e-4)
    parser.add_argument('--lr_c', type=float, default=1e-2)
    parser.add_argument('--train_a_itrs', type=int, default=10)
    parser.add_argument('--train_c_itrs', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=500)
    parser.add_argument('--eval', type=str, default='')
    parser.add_argument('--play_speed', type=float, default=1)
    parser.add_argument('--num_procs', type=int, default=1)
    args = parser.parse_args()

    run_experiments(args)
        