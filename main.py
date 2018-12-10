# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: main.py
# Description: entry file


from mpi4py import MPI
import numpy as np
import os
import subprocess
import sys
import time
import argparse

from param_server import ParamServer
from node import Node

from communicator import WorkerComm
from communicator import MasterComm
from tinynn.utils.timing import Timer

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


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


def main():
    if RANK == 0:
        print('Master started. %d processes.' % SIZE)
        Master(COMM, RANK).run()
    else:
        print('Worker-%d started. %d processes.' % (RANK, SIZE))
        Worker(COMM, RANK).run()


class Master(object):

    def __init__(self, comm, rank):
        self.comm = MasterComm(comm)
        self.param_server = ParamServer()
        self.rank = rank

    def run(self):
        i = 0
        timer = {
            'distribute': Timer('distribute'),
            'gather': Timer('gather'),
            'update': Timer('update')
        }
        while True:
            i += 1
            params = self.param_server.get_params
            # distribute contents
            timer['distribute'].start()
            self.comm.distribute(params.copy())
            timer['distribute'].pause()

            # gather results
            timer['gather'].start()
            all_local_params = self.comm.gather()
            timer['gather'].pause()

            # update global params
            timer['update'].start()
            self.param_server.update(all_local_params)
            timer['update'].pause()

            print('---------------')
            print('{}-iteration'.format(i))
            self.param_server.evaluate()

            if i % 5 == 0:
                for t in timer.values():
                    t.report()


class Worker(object):

    def __init__(self, comm, rank):
        self.node = Node()
        self.comm = WorkerComm(comm)
        self.rank = rank

    def run(self):
        while True:
            # pull global params
            global_params = self.comm.pull_global_params()
            # local update
            self.node.update(global_params)
            local_params = self.node.get_params
            # push to parameter server
            self.comm.push_local_params(local_params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_workers', type=int, default=2)
    parser.add_argument('-t', '--num_trails', type=int, help='trials per worker', default=4)
    parser.add_argument('-s', '--seed', type=int, default=31, help='initial seed')

    global args
    args = parser.parse_args()
    assert args.num_workers > 0, 'Number of workers suppose to > 0.'
    mpi_run()
