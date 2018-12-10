# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: node.py
# Description: Node class


import numpy as np

from config import batch_size, learning_rate, dataset_dir

from tinynn.data_processor.dataset import MNIST
from tinynn.data_processor.data_iterator import BatchIterator
from tinynn.core.nn import NeuralNet
from tinynn.core.layers import Linear, ReLU
from tinynn.core.loss import CrossEntropyLoss
from tinynn.core.optimizer import Adam
from tinynn.core.evaluator import AccEvaluator
from tinynn.core.model import Model


def get_one_hot(targets, nb_classes):
    return np.eye(nb_classes)[np.array(targets).reshape(-1)]


class Node(object):

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
        dataset = {'train_x': train_x,
                   'train_y': train_y,
                   'valid_x': valid_x,
                   'valid_y': valid_y}

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
        self.nn_model.net.set_parameters(params)
        # local training
        self._train_one_epoch()

    def _train_one_epoch(self):
        iterator = BatchIterator(batch_size=batch_size)
        evaluator = AccEvaluator()

        for batch in iterator(self.dataset['train_x'], self.dataset['train_y']):
            pred = self.nn_model.forward(batch.inputs)
            loss, grads = self.nn_model.backward(pred, batch.targets)
            self.nn_model.apply_grad(grads)

        # # evaluate
        # self.nn_model.set_phase('TEST')
        # valid_pred = model.forward(self.valid_x)
        # valid_pred_idx = np.argmax(valid_pred, axis=1)
        # valid_y_idx = np.asarray(self.valid_y)
        # res = evaluator.eval(valid_pred_idx, valid_y_idx)
        # print(res)
        # self.nn_model.set_phase('TRAIN')

    @property
    def get_params(self):
        return self.nn_model.net.get_parameters()
