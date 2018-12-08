from mpi4py import MPI
import numpy as np
import time

from communicator import WorkerComm


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

class Node(object):

    def __init__(self):
        self.__local_params = None

    def update(self, params):
        update = np.ones_like(params)
        self.__local_params = params + update

    @property
    def get_params(self):
        return self.__local_params
