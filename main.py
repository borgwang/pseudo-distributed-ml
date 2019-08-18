# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: main.py
# Description: entry file

import os
import sys
sys.path.append(os.path.abspath("./tinynn"))

from mpi4py import MPI
import subprocess
import argparse

from param_server import MAParamServer
from node import MANode

from communicator import WorkerComm
from communicator import MasterComm
from tinynn.utils.timer import Timer

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


def mpi_run():
    if os.getenv("IN_MPI") is None:
        # fork processes
        env = os.environ.copy()
        env.update(IN_MPI="1")
        mpi_cmd = ["mpirun", "-np", str(args.num_workers + 1)]
        script = [sys.executable, "-u"] + sys.argv
        cmd = mpi_cmd + script
        print("RUNNING: %s" % (" ".join(cmd)))
        subprocess.check_call(cmd, env=env)
        sys.exit()  # admin process exit
    else:
        main()


def main():
    if RANK == 0:
        print("Master started. %d processes." % SIZE)
        Master(COMM, RANK).run()
    else:
        print("Worker-%d started. %d processes." % (RANK, SIZE))
        Worker(COMM, RANK).run()


class Master(object):

    def __init__(self, comm, rank):
        self.comm = MasterComm(comm)
        self.param_server = MAParamServer()
        self.rank = rank

    def run(self):
        i = 0
        timer = {
            "iter": Timer("iter"),
            "distribute": Timer("distribute"),
            "gather": Timer("gather"),
            "update": Timer("update")
        }
        # while True:
        for _ in range(args.num_epochs):
            i += 1
            timer["iter"].start()
            params = self.param_server.get_params()

            # distribute contents
            timer["distribute"].start()
            self.comm.distribute(params.copy())
            timer["distribute"].pause()

            # gather results
            timer["gather"].start()
            local_results = self.comm.gather()
            timer["gather"].pause()

            # update global params
            timer["update"].start()
            self.param_server.update(local_results)
            timer["update"].pause()

            print("---------------")
            print("{}-iteration".format(i))
            self.param_server.evaluate()

            timer["iter"].pause()
            if i % 2 == 0:
                for t in timer.values():
                    t.report()
        print("reach max num_epochs {}".format(args.num_epochs))
        print("exit")


class Worker(object):

    def __init__(self, comm, rank):
        self.node = MANode()
        self.comm = WorkerComm(comm)
        self.rank = rank

    def run(self):
        # while True:
        for _ in range(args.num_epochs):
            # pull global params
            global_params = self.comm.pull_global_params()
            # local update
            self.node.update(global_params)

            local_results = self.node.get_results()

            # push to parameter server
            self.comm.push_local_results(local_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_workers", type=int, default=4)
    parser.add_argument("-s", "--seed", type=int, default=31, help="initial seed")
    parser.add_argument("--num_epochs", type=int, default=10)
    global args
    args = parser.parse_args()
    assert args.num_workers > 0, "Number of workers suppose to > 0."
    mpi_run()
