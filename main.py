from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble, neighbors, tree
from sklearn import base
import sklearn.linear_model
import random
import numpy as np
from sklearn.model_selection import KFold
import copy
import pandas as pd
import itertools
import gc
import os
import pickle

num_of_models = 5


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class Builder():
    def __init__(self, max_num_of_stacks = 10, mutations_per_model = 5, x=None, y = None, serialize_dir = ''):
        stacks = [Stack(gen=0, x=x, y=y) for _ in range(max_num_of_stacks)]
        save_dir = create_directory(serialize_dir + 'model_stacking_save_dir')
        score_dicts = []

        gen_count = 0
        while True:
            gen_count += 1
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)

            clf = ensemble.RandomForestClassifier(n_estimators=100)
            clf.fit(train_x, train_y)
            single_rf_score = clf.score(test_x, test_y)
            clf = ensemble.GradientBoostingClassifier()
            clf.fit(train_x, train_y)
            single_gbm_score = clf.score(test_x, test_y)

            print('single gbm:', single_gbm_score)
            print('single rf:', single_rf_score)
            gen_score_dict = {'single_gbm_score':single_gbm_score,
                              'single_rf_score':single_rf_score}


            next_gen_stacks = []
            for i in stacks:
                next_gen_stacks.append(i)
                for j in range(mutations_per_model):
                    next_gen_stacks.append(i.generate_mutated_model())

            stacks = next_gen_stacks
            del next_gen_stacks
            gc.collect()

            for i in stacks:
                i.load_models()
                i.train(train_x, train_y)
                i.score(test_x, test_y)
                i.del_models()

            stacks.sort(key=lambda i: i.last_tested_score, reverse=True)

            stacks = stacks[:max_num_of_stacks]
            print('\n')
            for i in stacks:
                print(gen_count, i.gen, i.last_tested_score, i.get_model_count())
            print('\n')

            with open(save_dir + '/' + str(gen_count) + '.plk', 'wb') as infile:
                pickle.dump(stacks[0], infile)

            gen_score_dict.update({'gen_count':gen_count,
                           'top_stack_gen':stacks[0].gen,
                           'top_stack_score': stacks[0].last_tested_score,
                           'top_stack_model_count': stacks[0].get_model_count()})
            score_dicts.append(gen_score_dict)
            df = pd.DataFrame.from_dict(score_dicts)
            df.to_csv('stack_results.csv')




class Stack():
    def __init__(self, max_layers = 3, max_models = 50, x = None, y = None, gen = 0):
        self.gen = 0
        self.max_layers = max_layers
        self.max_models = max_models
        #self.output_model = None
        self.model_layers = [[] for _ in range(max_layers + 1)]
        self.min_input_layer_choice = 1
        self.input_layer = [Node(0, i) for i in range(x.shape[1])]
        self.output_node = Node(max_layers + 1, 0)
        self.generate_first_model()
        self.last_tested_score = 0

    def del_models(self):
        for i in self.get_model_list():
            i.del_model()

    def load_models(self):
        for i in self.get_model_list():
            i.load_model()


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
        # if model_index == 5:
        #     return KNeighborsClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)


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
        changing_model = stack_copy.select_random_model()
        layer_to_create_model_on = changing_model.l_id
        valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)

        # for i in valid_inputs:
        #     print(1, i in self.input_layer)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
        changing_model.reset_model(model_inputs)


        '''
        reset final model
        '''
        changing_model = stack_copy.model_layers[-1][0]
        layer_to_create_model_on = changing_model.l_id
        valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
        changing_model.reset_model(model_inputs)

        stack_copy.gen = self.gen + 1
        return stack_copy

    def train(self, x, y):
        '''
        Given data and labels this splits it into folds and iteratively trains each layer
        on the outputs of previous models which predict a fold it has not trained on.
        '''

        #print(self.gen, len(self.input_layer))

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

        self.last_tested_score = self.model_layers[-1][0].score(x, y)
        return self.last_tested_score



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
        self.clf = base.BaseEstimator()

        self.l_id = l_id
        self.input_nodes = input_nodes
        self.output_node = output_node

        self.hyperparam_ranges = dict()
        self.set_hyperparameters = dict()


    def set_random_hyparameters(self):
        self.set_hyperparameters = self.get_random_hyperparameters()

    def del_model(self):
        del self.clf
        gc.collect()

    def load_model(self):
        pass

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

    def reset_model(self, input_nodes = None):
        self.input_nodes = input_nodes
        self.set_random_hyparameters()

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
        self.set_hyperparameters = dict()
        self.clf = ensemble.AdaBoostClassifier(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  ensemble.AdaBoostClassifier(**self.set_hyperparameters)


class RandomForestClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)

        self.hyperparam_ranges = {'n_estimators':[10, 200],
                                  'criterion':['gini', 'entropy'],
                                  'max_features':['sqrt', 'log2', 'auto'],
                                  'max_depth':[5, 50],
                                  'min_samples_split':[2,10],
                                  'min_samples_leaf':[1,5],
                                  'n_jobs':[-1, -1, -1]}
        self.set_hyperparameters = dict()
        self.clf = ensemble.RandomForestClassifier(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  ensemble.RandomForestClassifier(**self.set_hyperparameters)

class ExtraTreesClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'n_estimators':[10, 200],
                                  'criterion':['gini', 'entropy'],
                                  'max_features':['sqrt', 'log2', 'auto'],
                                  'max_depth':[5, 50],
                                  'min_samples_split':[2,10],
                                  'min_samples_leaf':[1,5],
                                  'n_jobs': [-1, -1, -1]}
        self.set_hyperparameters = dict()
        self.clf = ensemble.ExtraTreesClassifier(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  ensemble.ExtraTreesClassifier(**self.set_hyperparameters)


class GradientBoostingClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'loss':['deviance', 'exponential'],
                                  'learning_rate':[.01,.5],
                                  'n_estimators':[10, 200],
                                  'max_depth':[3, 10],
                                  'criterion':['friedman_mse', 'mse', 'mae'],
                                  'max_features':['sqrt', 'log2', 'auto']}
        self.set_hyperparameters = dict()
        self.clf = ensemble.GradientBoostingClassifier(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  ensemble.GradientBoostingClassifier(**self.set_hyperparameters)


class DecisionTreeClassifierModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'criterion':['gini', 'entropy'],
                                  'max_features': ['sqrt', 'log2', 'auto'],
                                  'max_depth': [5, 50],
                                  'min_samples_split': [2, 10],
                                  'min_samples_leaf': [1, 5]
                                  }
        self.set_hyperparameters = dict()
        self.clf = tree.DecisionTreeClassifier(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  tree.DecisionTreeClassifier(**self.set_hyperparameters)

#
# class KNeighborsClassifierModel(Model):
#
#     def __init__(self, l_id, input_nodes = None, output_node = None):
#         super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
#         self.hyperparam_ranges = {'n_neighbors':[3,20],
#                                   'weights':['uniform', 'distance'],
#                                   'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute'],
#                                   'leaf_size':[10, 100],
#                                   'n_jobs': [-1, -1, -1]}
#         self.clf = neighbors.KNeighborsClassifier(**self.get_random_hyperparameters())


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
    df = pd.read_csv('income.csv', nrows = 1000,
                     names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                            'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'y'])
    le = sklearn.preprocessing.LabelEncoder()

    df = df.applymap(lambda x: -1 if x == '?' else x)

    for i in ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'y']:
        df[i] = le.fit_transform(df[i])

    y_df = df[df.columns[-1]]
    x_df = df[df.columns[:-1]]
    x = x_df.as_matrix()
    y = y_df.as_matrix()
    Builder(x=x, y=y)

