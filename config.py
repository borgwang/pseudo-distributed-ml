# Author: borgwang <borgwang@126.com>
# Date: 2018-12-10
#
# Filename: config.py
# Description: nerual net architecture


from tinynn.core.layers import Linear, Sigmoid, ReLU


nn_architecture = [
    Linear(num_in=784, num_out=200),
    ReLU(),
    Linear(num_in=200, num_out=100),
    ReLU(),
    Linear(num_in=100, num_out=60),
    ReLU(),
    Linear(num_in=60, num_out=30),
    ReLU(),
    Linear(num_in=30, num_out=10)
]

dataset_dir = './data/'
