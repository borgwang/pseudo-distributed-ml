from mpi4py import MPI
import numpy as np

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


class Communicator(object):
    ''' A class that manage communication between master and workers. '''

    def __init__(self, comm):
        self._comm = comm

    @staticmethod
    def decode_packet(packet):
        contents = np.reshape(packet, (3, 5))
        return contents

    @staticmethod
    def encode_packet(contents):
        packet = np.ravel(contents)
        return packet


class MasterComm(Communicator):
    def __init__(self, comm):
        Communicator.__init__(self, comm)
        self.worker_ids = np.arange(1, SIZE)

    def distribute(self, global_params):
        packet = self.encode_packet(global_params)
        for worker_id in self.worker_ids:
            self._comm.Send(packet, dest=worker_id)

    def gather(self):
        packet = np.empty(15, dtype=float)
        results = list()
        for worker_id in self.worker_ids:
            self._comm.Recv(packet, source=worker_id)
            local_param = self.decode_packet(packet)
            results.append(local_param)
        return results


class WorkerComm(Communicator):
    def __init__(self, comm):
        Communicator.__init__(self, comm)
        self.master_id = 0

    def pull_global_params(self):
        packet = np.empty(shape=15, dtype=float)
        self._comm.Recv(packet, source=self.master_id)
        parmas = self.decode_packet(packet)
        return parmas

    def push_local_params(self, params):
        packet = self.encode_packet(params)
        self._comm.Send(packet, dest=self.master_id)
