from mpi4py import MPI
import numpy as np
import os
import subprocess
import sys
import time
import argparse

from master import Master
from worker import Worker


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def main():
    if RANK == 0:
        print('Master started. %d processes.' % SIZE)
        Master(COMM, RANK).run()
    else:
        print('Worker-%d started. %d processes.' % (RANK, SIZE))
        Worker(COMM, RANK).run()


def mpi_run():
    if os.getenv('IN_MPI') is None:
        # fork processes
        env = os.environ.copy()
        env.update(IN_MPI='1')
        mpi_cmd = ['mpirun', '-np', str(args.num_workers + 1)]
        script = [sys.executable, '-u'] + sys.argv
        cmd = mpi_cmd + script
        print('RUNNING: %s' % (' '.join(cmd)))
        subprocess.check_call(cmd, env=env)
        sys.exit()  # admin process exit
    else:
        main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_workers', type=int, default=2)
    parser.add_argument('-t', '--num_trails', type=int, help='trials per worker', default=4)
    parser.add_argument('-s', '--seed', type=int, default=111, help='initial seed')

    global args
    args = parser.parse_args()
    assert args.num_workers > 0, 'Number of workers suppose to > 0.'
    mpi_run()
