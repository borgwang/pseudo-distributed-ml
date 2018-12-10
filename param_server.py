# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: param_server.py
# Description: ParamServer class


import copy
import numpy as np

from tinynn.data_processor.dataset import MNIST
from tinynn.data_processor.data_iterator import BatchIterator
from tinynn.core.nn import NeuralNet
from tinynn.core.layers import Linear, ReLU
from tinynn.core.loss import CrossEntropyLoss
from tinynn.core.optimizer import Adam
from tinynn.core.evaluator import AccEvaluator
from tinynn.core.model import Model

from config import dataset_dir, architecture
from communicator import decode_packet, encode_packet


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class ParamServer(object):

    def __init__(self):
        self.test_x, self.test_y = self.load_test_data()
        self.nn_model = self.init_nn_model()

    @staticmethod
    def load_test_data():
        mnist = MNIST(dataset_dir)
        test_x, test_y = mnist.test_data
        test_y = get_one_hot(test_y, 10)
        return test_x, test_y

    @staticmethod
    def init_nn_model():
        nn_model = Model(
            net=NeuralNet([
                    Linear(num_in=784, num_out=100),
                    ReLU(),
                    Linear(num_in=100, num_out=10)]),
            loss_fn=CrossEntropyLoss(),
            optimizer=Adam())
        nn_model.initialize()
        nn_model.set_phase('TEST')
        return nn_model

    def update(self, all_local_results):
        # merge parameters or updates
        merged = list()
        for local in all_local_results:
            merged.append(encode_packet(local))
        merged = np.mean(merged, axis=0)

        # add update to parameters
        merged += encode_packet(self.get_params)
        merged = decode_packet(merged)

        # update global parameters
        self.set_params(merged)

    def evaluate(self):
        test_preds = self.nn_model.forward(self.test_x)
        test_pred_idx = np.argmax(test_preds, axis=1)
        test_y_idx = np.argmax(self.test_y, axis=1)
        res = AccEvaluator().eval(test_pred_idx, test_y_idx)
        print(res)

    @property
    def get_params(self):
        return copy.deepcopy(self.nn_model.net.get_parameters())

    def set_params(self, params):
        self.nn_model.net.set_parameters(params)
