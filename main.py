from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.linear_model
import random
import numpy as np
from sklearn.model_selection import KFold
import copy


class Builder():
    def __init__(self):
        pass


class Stack():
    def __init__(self, max_layers = 7, max_models = 50, x = None, y = None):
        self.max_layers = max_layers
        self.max_models = max_models
        self.input_layers = [Input_Layer(x)]
        self.model_layers = []
        self.min_input_layer_choice = 1

    #gen 0 model creation
    def generate_first_models(self, x, y):
        self.input_layer = Input_Layer(x)

        for i in range(self.max_layers):
            self.model_layers.append([])
            next_model = Model()
            next_model_input_layers = []
            for i in self.input_layers:
                if len(i.valid_indexes) > self.min_input_layer_choice:
                    next_model_input_size = random.randint(self.min_input_layer_choice, len(i.valid_indexes))
                    next_model_input_layers.append(random.sample(i.valid_indexes, next_model_input_size))
                else:
                    next_model_input_layers.append(i.valid_indexes)
            next_model.set_input_layers(next_model_input_layers[-1])
            self.model_layers[-1].append(next_model)
        final_model = Model()

        final_input = [[] for i in self.input_layers]
        final_input.append(self.input_layers[-1])
        final_model.set_input_layers(final_input)
        self.model_layers.append(final_model)



    def generate_mutated_model(self):
        '''
        if model = maximum num of models and a model is elegible for deletion, delete random eligible model

        if model < maximum num of models, add random model to random layer

        pick random model, reset hyperparameters
        :return:
        '''

        stack_copy = copy.deepcopy(self)



        pass

    def get_indexes_of_models_elegible_for_deletion(self):
        for i in range(self.max_layers, 0):
            pass


#generalized model wrapper to be able to use same functions on different implementations of models
class Model():

    def __init__(self):
        self.clf = ensemble.AdaBoostClassifier()

    def set_input_layers(self, input_layers):
        pass

    def fit(self, x, y):
        self.clf.fit(x, y)

    def score(self, x, y):
        return

class Input_Layer():

    def __init__(self, x):
        self.valid_indexes = [i for i in range(x.shape(0))]

    def add_input(self):
        self.valid_indexes.append(max(self.valid_indexes) + 1)