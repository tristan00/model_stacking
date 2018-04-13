from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import ensemble
import sklearn.linear_model
import random
import numpy as np
from sklearn.model_selection import KFold
import copy


class Builder():
    def __init__(self, max_num_of_stacks = 10, num_of_mutations = 10, x=None, y = None):
        gen_0_stacks = [Stack() for _ in range(max_num_of_stacks)]


class Stack():
    def __init__(self, max_layers = 7, max_models = 50, x = None, y = None):
        self.max_layers = max_layers
        self.max_models = max_models
        self.input_layers = [Input_Layer(x)]
        self.model_layers = []
        self.min_input_layer_choice = 1


    def generate_first_models(self, x, y):
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
            self.input_layers[i+1].add_input()
            self.model_layers[-1].append(next_model)
        final_model = Model()

        final_input = [[] for i in self.input_layers]
        final_input.append(self.input_layers[-1])
        final_model.set_input_layers(final_input)
        self.model_layers.append(final_model)



    def generate_mutated_model(self):
        '''
        if model = maximum num of models and a model is elegible for deletion, delete random eligible model, not implemented
        if model < maximum num of models, add random model to random layer
        pick random model, reset hyperparameters
        reset final model
        :return:
        '''

        stack_copy = copy.deepcopy(self)
        if stack_copy.get_model_count() < stack_copy.max_models:
            layer = random.randint(0, stack_copy.max_layers - 1)

            added_model = Model(output_index = self.input_layers[layer] + 1)
            next_model_input_layers = []
            for i in range(layer):
                if len(self.input_layers[i].valid_indexes) > self.min_input_layer_choice:
                    next_model_input_size = random.randint(self.min_input_layer_choice, len(self.input_layers[i].valid_indexes))
                    next_model_input_layers.append(random.sample(self.input_layers[i].valid_indexes, next_model_input_size))
                else:
                    next_model_input_layers.append(self.input_layers[i].valid_indexes)
            added_model.set_input_layers(next_model_input_layers[-1])
            self.input_layers[layer + 1].add_input()
            self.model_layers[layer].append(added_model)
        pass


    def get_indexes_of_models_elegible_for_deletion(self):
        for i in range(0, self.max_layers):
            pass


    def get_model_count(self):
        return sum([len(i) for i in self.model_layers])


#generalized model wrapper to be able to use same functions on different implementations of models
class Model():

    def __init__(self, output_index = 0):
        self.clf = ensemble.AdaBoostClassifier()
        self.output_index = output_index


    def set_random_hyperparameters(self):
        pass


    def set_input_layers(self, input_layers):
        pass


    def fit(self, x, y):
        self.clf.fit(x, y)


    def score(self, x, y):
        return


class Input_Layer():


    def __init__(self, x, is_first_layer):
        self.input_nodes = []
        if is_first_layer:
            self.input_nodes = [Node(i) for i in range(x.shape[1])]


    def add_node(self, n):
        self.input_nodes.append(n)


    def load(self, x):
        for i, n in zip(x.T, self.input_nodes):
            n.load(i)


    def dump(self, n_ids):
        called_input_nodes = [i for i in self.input_nodes.n_id if i.n_id in n_ids]
        called_input_nodes = sorted(called_input_nodes, key= lambda x: x.n_id)
        return np.hstack([i.dump() for i in called_input_nodes])


class Node():

    def __init__(self, n_id):
        self.n_id = n_id
        self.value = None

    def load(self, x):
        self.value = x

    def dump(self):
        return self.value