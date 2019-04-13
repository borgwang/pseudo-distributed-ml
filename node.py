# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: node.py
# Description: Node class

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

from config import batch_size, learning_rate, dataset_dir
from communicator import decode, encode


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class BaseNode(object):

    def __init__(self):
        self.nn_model = self.init_nn_model()
        self.dataset = self.load_dataset()

    @staticmethod
    def load_dataset():
        mnist = MNIST(dataset_dir)
        train_x, train_y = mnist.train_data
        valid_x, valid_y = mnist.valid_data
        train_y = get_one_hot(train_y, 10)
        valid_y = get_one_hot(valid_y, 10)
        dataset = {'train_x': train_x, 'train_y': train_y,
                   'valid_x': valid_x, 'valid_y': valid_y}

        return dataset

    @staticmethod
    def init_nn_model():
        nn_model = Model(
            net=NeuralNet([
                Linear(num_in=784, num_out=100),
                ReLU(),
                Linear(num_in=100, num_out=10)]),
            loss_fn=CrossEntropyLoss(),
            optimizer=Adam(learning_rate))
        nn_model.initialize()
        nn_model.set_phase('TRAIN')
        return nn_model

    def update(self, params):
        raise NotImplementedError

    def get_results(self):
        raise NotImplementedError

    def _train_one_epoch(self):
        start_params = self.get_params()

        iterator = BatchIterator(batch_size=batch_size)
        evaluator = AccEvaluator()

        for batch in iterator(self.dataset['train_x'], self.dataset['train_y']):
            pred = self.nn_model.forward(batch.inputs)
            loss, grads = self.nn_model.backward(pred, batch.targets)
            self.nn_model.apply_grad(grads)

        end_params = self.get_params()
        updates = decode(
            encode(end_params) - encode(start_params))
        return updates

    def get_params(self):
        return copy.deepcopy(self.nn_model.net.get_parameters())

    def set_params(self, params):
        self.nn_model.net.set_parameters(params)


class MANode(BaseNode):
    '''
    Model Averaging Node
    '''
    def get_results(self):
        return self.get_params()

    def update(self, params):
        # force sync with global parameters
        self.set_params(params)
        # local training
        self._train_one_epoch()


class SSGDNode(BaseNode):
    '''
    Synchronous SGD
    '''
    def __init__(self):
        super(SSGDNode, self).__init__()
        self.__updates = None

    def get_results(self):
        return self.__updates

    def update(self, params):
        # force sync with global parameters
        self.set_params(params)
        # local training
        updates = self._train_one_epoch()
        self.__updates = updates


class EASGDNode(BaseNode):
    '''
    Elastic Averaging SGD
    See https://arxiv.org/abs/1412.6651 for details
    '''
    def get_results(self):
        return self.get_params()

    def update(self, params):
        # elastic update parameters
        self._elastic_update(params)
        # local training
        self._train_one_epoch()

    def _elastic_update(self, params):
        # limit the divergence local model and global model
        grads = 2 * (encode(self.get_params()) - encode(params))
        grads *= 0.005
        grads = decode(grads)
        self.nn_model.apply_grad(grads)


class BMUFNode(BaseNode):
    '''
    Block-wise Model Update Filtering Node
    see https://www.microsoft.com/en-us/research/wp-content/uploads/2016/08/0005880.pdf
    '''
    def get_results(self):
        return self.get_params()

    def update(self, params):
        # force sync with global parameters
        self.set_params(params)
        # local training
        self._train_one_epoch()
