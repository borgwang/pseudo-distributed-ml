from mpi4py import MPI
import numpy as np
import time

from communicator import MasterComm


class Master(object):

    def __init__(self, comm, rank):
        self.comm = MasterComm(comm)
        self.param_server = ParamServer()
        self.rank = rank

    def run(self):
        start_time = int(time.time())
        i = 0
        while True:
            i += 1
            params = self.param_server.get_params
            time.sleep(1)
            print(params)
            # distribute contents
            self.comm.distribute(params.copy())
            # gather results
            all_local_params = self.comm.gather()
            # update global params
            self.param_server.update(all_local_params)
            curr_time = int(time.time()) - start_time


class ParamServer(object):

    def __init__(self):
        # self._params = np.random.normal(size=(3, 5))
        self._params = np.ones(shape=(3, 5), dtype=float)

    def update(self, all_local_params):
        self._params = np.sum(all_local_params, axis=0)

    @property
    def get_params(self):
        return self._params
