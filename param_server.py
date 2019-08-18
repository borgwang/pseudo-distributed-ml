# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: param_server.py
# Description: ParamServer class


import copy

import numpy as np

from communicator import decode
from communicator import encode
from config import dataset_dir
from tinynn.core.evaluator import AccEvaluator
from tinynn.core.layers import Dense
from tinynn.core.layers import ReLU
from tinynn.core.losses import SoftmaxCrossEntropyLoss
from tinynn.core.model import Model
from tinynn.core.nn import Net
from tinynn.core.optimizer import Adam
from utils import prepare_dataset


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class BaseParamServer(object):

    def __init__(self):
        self.test_x, self.test_y = self.load_test_data()
        self.input_dim = self.test_x.shape[1]
        self.output_dim = self.test_y.shape[1]
        self.nn_model = self.init_nn_model()
        self.momentum = None

    @staticmethod
    def load_test_data():
        train_set, valid_set, test_set = prepare_dataset(dataset_dir)
        test_x, test_y = test_set
        test_y = get_one_hot(test_y, 10)
        return test_x, test_y

    def init_nn_model(self):
        hidden_units = 100
        net = Net([
            Dense(num_out=hidden_units, num_in=self.input_dim),
            ReLU(),
            Dense(num_out=self.output_dim, num_in=hidden_units)])
        nn_model = Model(net, loss=SoftmaxCrossEntropyLoss(),
                         optimizer=Adam())
        nn_model.set_phase("TEST")
        return nn_model

    def update(self, local_results):
        raise NotImplementedError

    def evaluate(self):
        test_preds = self.nn_model.forward(self.test_x)
        test_pred_idx = np.argmax(test_preds, axis=1)
        test_y_idx = np.argmax(self.test_y, axis=1)
        res = AccEvaluator().evaluate(test_pred_idx, test_y_idx)
        print(res)

    def get_params(self):
        return copy.deepcopy(self.nn_model.net.get_parameters())

    def set_params(self, params):
        self.nn_model.net.set_parameters(params)


class MAParamServer(BaseParamServer):
    """
    Model Averaging Parameter Server
    """

    def update(self, local_params):
        # averaging parameters
        params = list()
        for local in local_params:
            params.append(encode(local))
        average_params = decode(np.mean(params, axis=0))

        # update global parameters
        self.set_params(average_params)


class SSGDParamServer(BaseParamServer):
    """
    Synchronous SGD Parameter Server
    """

    def update(self, local_grads):
        # averaging gradients
        grads = list()
        for local in local_grads:
            grads.append(encode(local))
        average_grads = decode(np.mean(grads, axis=0))

        # add to parameters
        average_params = average_grads + encode(self.get_params())

        # update global parameters
        self.set_params(average_params)


class EASGDParamServer(BaseParamServer):
    """
    Elastic Averaging SGD Parameter Server
    See https://arxiv.org/abs/1412.6651 for details
    """

    def __init__(self):
        super(EASGDParamServer, self).__init__()
        self._param_momentum = None

    def update(self, local_params):
        # averaging local parameters
        params = list()
        for local in local_params:
            params.append(encode(local))
        average_params = np.mean(params, axis=0)

        if self._param_momentum is None:
            params = encode(self.get_params())
        else:
            params = 0.5 * self._param_momentum + 0.5 * average_params

        # update momentum
        self._param_momentum = params

        # update global parameters
        self.set_params(params)


class BMUFParamServer(BaseParamServer):
    """
    Block-wise Model Update Filtering ParamServer
    see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    """

    def __init__(self):
        super(BMUFParamServer, self).__init__()
        self._grad_momentum = None

    def update(self, local_params):
        # averaging local parameters
        params = list()
        for local in local_params:
            params.append(encode(local))
        average_params = np.mean(params, axis=0)

        grads = average_params - encode(self.get_params())
        if self._grad_momentum is None:
            updates = grads
        else:
            updates = 0.5 * self._grad_momentum + 0.5 * grads
        self._grad_momentum = updates
        params = updates + encode(self.get_params())

        # update global parameters
        self.set_params(params)
