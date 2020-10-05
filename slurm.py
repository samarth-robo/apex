#!/usr/bin/env python
import os
import sys
import random
import string
import argparse

osp = os.path


def random_filename(N=5):
    return ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(N))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('-c', default=1, type=int)
    parser.add_argument('-g', default=0, type=int)
    parser.add_argument('-w', default=None)
    parser.add_argument('--qos', default='normal')
    parser.add_argument('cmd', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    cmd = ' '.join(args.cmd)
    print(cmd)

    if not osp.isdir('logs'):
        os.mkdir('logs')
    name = osp.join('logs', random_filename())
    job_filename = '{:s}.slurm'.format(name)
    with open(job_filename, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        fh.writelines("#SBATCH --output={:s}.out\n".format(name))
        print('stdout in {:s}.out'.format(name))
        fh.writelines("#SBATCH --error={:s}.err\n".format(name))
        print('stderr in {:s}.err'.format(name))
        fh.writelines('#SBATCH -p {:s}\n'.format(args.p))
        fh.writelines('#SBATCH -c {:d}\n'.format(args.c))
        if args.w is not None:
          print('Requesting specifically node {:s}'.format(args.w))
          fh.writelines('#SBATCH -w {:s}\n'.format(args.w))
        fh.writelines('#SBATCH --qos {:s}\n'.format(args.qos))
        if args.g > 0:
            fh.writelines('#SBATCH --gres=gpu:{:d}\n'.format(args.g))
        # fh.writelines('#SBATCH --get-user-env\n')
        fh.writelines('eval "$(conda shell.bash hook)"\n')
        fh.writelines('conda activate apex\n')
        fh.writelines('{:s}\n'.format(cmd))
    os.system('sbatch {:s}'.format(job_filename))
