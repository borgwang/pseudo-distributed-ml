from mpi4py import MPI
import numpy as np

from config import architecture


def decode(packet):
    # decode 1-d array to parameters (dict)
    assert packet is not None, "Invalid packet."
    contents = list()
    pointer = 0
    for s in architecture:
        layer = dict()
        if len(s) < 1:
            contents.append(layer)
            continue
        layer["w"] = packet[pointer:pointer + s[0] * s[1]].reshape((s[0], s[1]))
        pointer += s[0] * s[1]
        layer["b"] = packet[pointer: pointer + s[1]].reshape((1, s[1]))
        pointer += s[1]
        contents.append(layer)
    return contents


def encode(contents):
    # encode parameters (dict) to 1-d array
    assert len(contents) > 0, "Invalid parameters."
    all_params = list()
    for l in contents:
        if len(l) == 0:
            continue
        all_params.append(np.ravel(l["w"]))
        all_params.append(np.ravel(l["b"]))
    packet = np.concatenate(all_params)
    return packet


class Communicator(object):
    """ A class that manage communication between master and workers. """

    def __init__(self, comm):
        self._comm = comm
        self.packet_size = 0
        for s in architecture:
            if len(s) > 0:
                self.packet_size += (s[0] * s[1] + s[1])


class MasterComm(Communicator):

    def __init__(self, comm):
        Communicator.__init__(self, comm)
        self.worker_ids = np.arange(1, comm.Get_size())

    def distribute(self, global_params):
        packet = encode(global_params)
        for worker_id in self.worker_ids:
            self._comm.Send(packet, dest=worker_id)

    def gather(self):
        packet = np.empty(self.packet_size, dtype=float)
        results = list()
        for worker_id in self.worker_ids:
            self._comm.Recv(packet, source=worker_id)
            local_param = decode(packet)
            results.append(local_param)
        return results


class WorkerComm(Communicator):

    def __init__(self, comm):
        Communicator.__init__(self, comm)
        self.master_id = 0

    def pull_global_params(self):
        packet = np.empty(shape=self.packet_size, dtype=float)
        self._comm.Recv(packet, source=self.master_id)
        parmas = decode(packet)
        return parmas

    def push_local_results(self, params):
        packet = encode(params)
        self._comm.Send(packet, dest=self.master_id)
