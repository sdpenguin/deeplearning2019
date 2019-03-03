'''
 A set of useful libraries for deep learning operations in Keras.
 '''
import numpy as np
import math
import random

from matplotlib import pyplot as plt
from random import randint

def data_stats(train_data, test_data=None, request='data_split', p=False):
    '''Returns heuristic statistics about the data input.'''
    acceptable_inputs = ['data_split', 'data_shape', 'data_length']
    if not request in acceptable_inputs:
        raise ValueError(
            'You need to enter a value from: {}'.format(acceptable_inputs))
    to_print = []

    def switcher(input):
        switcher_out = {
            'data_split': (lambda train_data, test_data: len(train_data) / (len(train_data) + len(test_data))),
            'data_shape': (lambda train_data, test_data: np.shape(train_data[0])),
            'data_length': (lambda train_data, test_data: len(train_data) + len(test_data)),
        }
        return switcher_out.get(input)
    output = switcher(request)(train_data, test_data)
    if p:
        print("{}: {}".format(request, output))
    return output

def return_element(data, labels=None):
    ''' Returns a generator for the given list.
        If corresponding labels are given, then we return tuples.
    '''
    for i in range(len(data)):
        if labels:
            yield (data[i], labels[i])
        else:
            yield data[i]


def random_subset(N, n=1, seed=None):
    ''' Returns a random number of indexes n for an N dimensional vector.
        By default returns just 1 element.'''
    if seed:
        random.seed(seed)
    indexes = [randint(0, N-1) for i in range(n)]
    return indexes

def plot_examples(data_in, n=1, labels=None, scale=1, cmap="gray", seed=None):
    ''' Plots n examples from the data_in using imshow.
        If you supply corresponding labels they will also be plotted.
        Scale is a number used to scale the entire figure size.
        Specify cmap to plot the figures in colour.
        '''
    if len(data_in) < n **2:
        raise ValueError('Too few data elements to plot.')
    dims = math.floor(math.sqrt(n))
    indexes = random_subset(len(data_in), dims**2, seed)
    if labels is not None:
        elements = [(data_in[i], labels[i]) for i in indexes]
    else:
        elements = [data_in[i] for i in indexes]
    element_yielder = return_element(elements)
    if dims == 1:
        fig, axes = plt.subplots(1, figsize=(scale*dims, scale*dims))

        current_image = element_yielder.__next__()
        if labels is not None:
            axes.imshow(current_image[0], cmap=cmap)
            axes.set_title(current_image[1])
        else:
            axes.imshow(current_image, cmap=cmap)
        axes.set_xticks([])
        axes.set_yticks([])
        return
    else:
        fig, axes = plt.subplots(dims, dims, figsize=(scale*dims, scale*dims))

        for row in range(dims):
            for col in range(dims):
                current_image = element_yielder.__next__()
                if labels is not None:
                    axes[row,col].imshow(current_image[0], cmap=cmap)
                    axes[row,col].set_title(current_image[1])
                else:
                    axes[row,col].imshow(current_image, cmap=cmap)
                axes[row,col].set_xticks([])
                axes[row,col].set_yticks([])
                fig.subplots_adjust(hspace=0.5)