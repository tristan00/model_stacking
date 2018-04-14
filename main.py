from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import ensemble, neighbors, tree
import sklearn.linear_model
import random
import numpy as np
from sklearn.model_selection import KFold
import copy
import pandas as pd
import itertools

num_of_models = 6


class Builder():
    def __init__(self, max_num_of_stacks = 10, num_of_mutations = 10, x=None, y = None):
        stacks = [Stack(gen=0, x=x, y=y) for _ in range(max_num_of_stacks)]



        gen_count = 0
        while True:
            gen_count += 1
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)

            next_gen_stacks = []
            for i in stacks:
                next_gen_stacks.append(i)
                next_gen_stacks.append(i.generate_mutated_model())

            stacks = next_gen_stacks

            for i in stacks:
                i.train(train_x, train_y)

            stacks.sort(key=lambda i: i.score(test_x, test_y), reverse=True)

            stacks = stacks[:int(len(stacks)/2)]
            print('\n')
            for i in stacks:
                print(gen_count, i.gen, i.score(test_x, test_y), i.get_model_count())



class Stack():
    def __init__(self, max_layers = 7, max_models = 50, x = None, y = None, gen = 0):
        self.gen = 0
        self.max_layers = max_layers
        self.max_models = max_models
        #self.output_model = None
        self.model_layers = [[] for _ in range(max_layers + 1)]
        self.min_input_layer_choice = 1
        self.input_layer = [Node(0, i) for i in range(x.shape[1])]
        self.output_node = Node(max_layers + 1, 0)
        self.generate_first_model()


    def load_input_layer(self, x):
        '''
        loading input into layer 0
        '''
        for i, j in zip(x.T, self.get_input_layer_at_layer_l(0)):
            j.load(i)


    def get_input_layer_at_layer_l(self, l):
        return [i for i in self.input_layer if i.l_id == l]


    def get_valid_inputs_for_layer(self, l):
        if l == 0:
            return self.get_input_layer_at_layer_l(0)
        else:
            return self.get_input_layer_at_layer_l(0) + self.get_input_layer_at_layer_l(l)


    def generate_first_model(self):
        '''
        creates single model at the last layer
        '''
        valid_inputs = self.get_valid_inputs_for_layer(self.max_layers)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))

        self.model_layers[-1].append(self.get_random_model(len(self.model_layers), model_inputs, self.output_node))


    def get_next_n_id(self, l):
        if len([i.n_id for i in self.input_layer if i.l_id == l]) > 0:
            return max([i.n_id for i in self.input_layer if i.l_id == l]) + 1
        else:
            return 0


    #TODO: get random model from allowed set
    def get_random_model(self, l_id, input_nodes=None, output_node=None):
        model_index = random.randint(0, num_of_models -1)

        if model_index == 0:
            return AdaBoostClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        if model_index == 1:
            return RandomForestClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        if model_index == 2:
            return ExtraTreesClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        if model_index == 3:
            return GradientBoostingClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        if model_index == 4:
            return DecisionTreeClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        if model_index == 5:
            return KNeighborsClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)


    def generate_mutated_model(self):
        '''
        if model = maximum num of models and a model is elegible for deletion, delete random eligible model, not implemented
        if model < maximum num of models, add random model to random layer
        pick random model, reset hyperparameters
        reset final model
        :return:
        '''

        stack_copy = copy.deepcopy(self)


        '''
        create model if valid to do so
        '''
        if stack_copy.get_model_count() < stack_copy.max_models:
            layer_to_create_model_on = random.randint(0, stack_copy.max_layers - 1)
            valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
            model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
            model_output = Node(layer_to_create_model_on + 1, stack_copy.get_next_n_id(layer_to_create_model_on + 1))
            stack_copy.input_layer.append(model_output)
            stack_copy.model_layers[layer_to_create_model_on].append(stack_copy.get_random_model(layer_to_create_model_on, input_nodes=model_inputs, output_node=model_output))

        '''
        randomly reset a model
        '''
        # changing_model = self.select_random_model()
        # layer_to_create_model_on = changing_model.l_id
        # valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
        # model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
        # changing_model.reset_model(layer_to_create_model_on, model_inputs, changing_model.output_node)

        stack_copy.gen = self.gen + 1
        return stack_copy

    def train(self, x, y):
        '''
        Given data and labels this splits it into folds and iteratively trains each layer
        on the outputs of previous models which predict a fold it has not trained on.
        '''

        xs = np.array_split(x, 2)
        ys = np.array_split(y, 2)

        x1 = xs[0]
        x2 = xs[1]

        y1 = ys[0]
        y2 = ys[1]

        for l_count, l in enumerate(self.model_layers):
            if l_count % 2 == 1:
                train_x = x1
                train_y = y1
            else:
                train_x = x2
                train_y = y2

            self.load_input_layer(train_x)
            for prev_l_index in range(0, l_count):

                for m in self.model_layers[prev_l_index]:
                    m.predict(train_x, train_y)

            for m in l:
                m.fit(train_x,train_y)

    def score(self, x, y):
        self.load_input_layer(x)
        for l in self.model_layers:
            for m in l:
                m.predict(x, y)

        return self.model_layers[-1][0].score(x, y)



    def get_model_count(self):
        return sum([len(i) for i in self.model_layers])


    def get_model_list(self):
        model_list = []
        for i in self.model_layers:
            model_list.extend(i)
        return model_list

    def select_random_model(self):
        return random.choice(self.get_model_list())


class Model():
    '''
    Generalized model wrapper to allow different libraries of models to work the same way
    '''

    def __init__(self, l_id, input_nodes = None, output_node = None):
        self.clf = None

        self.l_id = l_id
        self.input_nodes = input_nodes
        self.output_node = output_node

        self.hyperparam_ranges = {}

    def get_random_hyperparameters(self):
        selected_hyper_parameters = dict()

        for i, j in self.hyperparam_ranges.items():
            if isinstance(j[0], int) and len(j) == 2:
                selected_hyper_parameters[i] = random.randint(j[0], j[1])
            elif isinstance(j[0], float):
                selected_hyper_parameters[i] = random.uniform(j[0],j[-1])
            else:
                selected_hyper_parameters[i] = random.choice(j)
        return selected_hyper_parameters

    def reset_model(self, l_id, input_nodes = None, output_node = None):
        self.__init__(l_id, input_nodes = input_nodes, output_node = output_node)

    def get_input(self):
        output_list = [i.dump() for i in self.input_nodes]
        output_array = np.vstack([i for i in output_list])
        return output_array.T

    def fit(self, x, y):
        input_x = self.get_input()
        self.clf.fit(input_x, y)

    def predict(self, x, y):
        input_x = self.get_input()
        self.output_node.load(self.clf.predict(input_x))

    def score(self, x, y):
        input_x = self.get_input()
        return self.clf.score(input_x, y)

class AdaBoostClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'n_estimators':[10, 200],
                                  'learning_rate':[.1,1],
                                  'algorithm':['SAMME', 'SAMME.R']}
        self.clf = ensemble.AdaBoostClassifier(**self.get_random_hyperparameters())



class RandomForestClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)

        self.hyperparam_ranges = {'n_estimators':[10, 200],
                                  'criterion':['gini', 'entropy'],
                                  'max_features':['sqrt', 'log2', 'auto'],
                                  'max_depth':[5, 50],
                                  'min_samples_split':[2,10],
                                  'min_samples_leaf':[1,5]}
        self.clf = ensemble.RandomForestClassifier(**self.get_random_hyperparameters())

class ExtraTreesClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'n_estimators':[10, 200],
                                  'criterion':['gini', 'entropy'],
                                  'max_features':['sqrt', 'log2', 'auto'],
                                  'max_depth':[5, 50],
                                  'min_samples_split':[2,10],
                                  'min_samples_leaf':[1,5]}
        self.clf = ensemble.ExtraTreesClassifier(**self.get_random_hyperparameters())


class GradientBoostingClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'loss':['deviance', 'exponential'],
                                  'learning_rate':[.01,.5],
                                  'n_estimators':[10, 200],
                                  'max_depth':[3, 10],
                                  'criterion':['friedman_mse', 'mse', 'mae'],
                                  'max_features':['sqrt', 'log2', 'auto']}
        self.clf = ensemble.GradientBoostingClassifier(**self.get_random_hyperparameters())


class DecisionTreeClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'criterion':['gini', 'entropy'],
                                  'max_features': ['sqrt', 'log2', 'auto'],
                                  'max_depth': [5, 50],
                                  'min_samples_split': [2, 10],
                                  'min_samples_leaf': [1, 5]
                                  }
        self.clf = tree.DecisionTreeClassifier(**self.get_random_hyperparameters())


class KNeighborsClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'n_neighbors':[3,20],
                                  'weights':['uniform', 'distance'],
                                  'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
                                  'leaf_size':[10, 100]}
        self.clf = neighbors.KNeighborsClassifier(**self.get_random_hyperparameters())


class Node():

    def __init__(self, l_id, n_id):
        self.n_id = n_id
        self.l_id = l_id
        self.value = None

    def load(self, x):
        self.value = x

    def dump(self):
        return self.value





if __name__ == '__main__':
    df = pd.read_csv('tumor_dataset.csv', header=None)
    df[1] = df[1].apply(lambda x: 1 if x=='M' else 0)
    df = df.drop([0], axis = 1)
    y_df = df[df.columns[0]]
    x_df = df[df.columns[1:]]
    x = x_df.as_matrix()
    y = y_df.as_matrix()

    Builder(x=x, y=y)

