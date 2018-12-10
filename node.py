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
from communicator import decode_packet, encode_packet


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class Node(object):

    def __init__(self):
        self.nn_model = self.init_nn_model()
        self.dataset = self.load_dataset()

        self.__updates = None

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
        # sync with global parameters
        self.set_params(params)
        # local training
        self._train_one_epoch()

    def _train_one_epoch(self):
        start_params = self.get_params

        iterator = BatchIterator(batch_size=batch_size)
        evaluator = AccEvaluator()

        for batch in iterator(self.dataset['train_x'], self.dataset['train_y']):
            pred = self.nn_model.forward(batch.inputs)
            loss, grads = self.nn_model.backward(pred, batch.targets)
            self.nn_model.apply_grad(grads)

        end_params = self.get_params
        self.__updates = decode_packet(
            encode_packet(end_params) - encode_packet(start_params))

    @property
    def get_params(self):
        return copy.deepcopy(self.nn_model.net.get_parameters())

    def set_params(self, params):
        self.nn_model.net.set_parameters(params)

    @property
    def get_updates(self):
        return self.__updates
