from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import ensemble, neighbors, tree
from sklearn.metrics import accuracy_score
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
from graphviz import Graph, Digraph
import xgboost as xgb
import lightgbm as lgb
import catboost
import traceback
import fastcluster
import hdbscan
from sklearn.cluster import MiniBatchKMeans

num_of_models = 5


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class Builder():
    def __init__(self, max_num_of_stacks = 5, mutations_per_model = 10, x=None, y = None, serialize_dir = '', max_generations = 100, errors = 'ignore'):
        stacks = [Stack(gen=0, x=x, y=y) for _ in range(max_num_of_stacks)]
        save_dir = create_directory(serialize_dir + 'model_stacking_save_dir')
        score_dicts = []

        gen_count = 0
        for _ in range(max_generations):
            gen_count += 1
            train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)

            clf = ensemble.RandomForestClassifier(n_jobs=-1)
            clf.fit(train_x, train_y)
            single_rf_score = clf.score(test_x, test_y)
            clf = ensemble.GradientBoostingClassifier()
            clf.fit(train_x, train_y)
            single_gbm_score = clf.score(test_x, test_y)


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

            failed_models = []
            for i in stacks:
                try:
                    i.load_models()
                    i.train(train_x, train_y)
                    i.score(test_x, test_y)
                    i.del_models()
                except:
                    traceback.print_exc()
                    failed_models.append(i)

            for i in failed_models:
                stacks.remove(i)

            stacks.sort(key=lambda i: i.last_tested_score, reverse=True)

            stacks = stacks[:max_num_of_stacks]
            print('\n')
            print(gen_count, 'single gbm:', single_gbm_score)
            print(gen_count, 'single rf:', single_rf_score)
            for i in stacks:
                print(gen_count, i.gen, i.last_tested_score, i.get_model_count(), str(id(i)))
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
    def __init__(self, max_layers = 4, max_models = 16, x = None, y = None, gen = 0):
        self.gen = 0
        self.max_layers = max_layers
        self.max_models = max_models
        #self.output_model = None
        self.model_layers = [[] for _ in range(max_layers)]
        self.min_input_layer_choice = 1
        self.input_layer = [Node(0, i) for i in range(x.shape[1])]
        self.output_node = Node(max_layers + 1, 0)
        self.generate_first_model()
        # self.generate_random_stack()
        self.last_tested_score = 0



    def generate_mutated_model(self, mutation_level = 'random'):
        '''
        if model = maximum num of models and a model is elegible for deletion, delete random eligible model, not implemented
        if model < maximum num of models, add random model to random layer
        pick random model, reset hyperparameters
        reset final model
        :return:
        '''

        stack_copy = copy.deepcopy(self)

        add_edge = random.randint(0, 1)
        remove_edge = random.randint(0, 1)
        reset_model = random.randint(0, 1)
        #create_model = random.randint(0, 1)
        create_model = 1
        del_model = random.randint(0, 1)




        if add_edge == 1:
            stack_copy.add_edge()

        if remove_edge == 1:
            stack_copy.remove_edge()


        '''
        remove model if valid to do so
        '''
        if stack_copy.get_model_count() >= stack_copy.max_models and del_model == 1:
            stack_copy.reset_model()


        '''
        create model if valid to do so
        '''
        if stack_copy.get_model_count() < stack_copy.max_models and create_model == 1:
            stack_copy.add_model()



        '''
        reset random model
        '''
        if reset_model == 1 and len(stack_copy.get_model_list()) > 1:
            stack_copy.reset_model()

        '''
        reset final model, possible change to model type
        '''
        changing_model = stack_copy.model_layers[-1][0]
        layer_to_create_model_on = changing_model.l_id
        valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
        new_model = stack_copy.get_random_model(layer_to_create_model_on, input_nodes=valid_inputs, output_node=changing_model.output_node)
        stack_copy.model_layers[-1][0] = new_model

        stack_copy.gen = self.gen + 1
        return stack_copy

    def add_edge(self):
        models = self.get_model_list()
        models.remove(self.model_layers[-1][0])
        model_to_modify = random.choice(models)
        valid_inputs = self.get_valid_inputs_for_layer(model_to_modify.l_id)
        valid_inputs = [i for i in valid_inputs if i not in model_to_modify.input_nodes]
        if len(valid_inputs) > 0:
            model_to_modify.input_nodes.append(random.choice(valid_inputs))


    def remove_edge(self):
        models = self.get_model_list()
        models.remove(self.model_layers[-1][0])
        model_to_modify = random.choice(models)

        if len(model_to_modify.input_nodes) > 2:
            model_to_modify.input_nodes.remove(random.choice(model_to_modify.input_nodes))


    def remove_model(self):
        model_list = self.get_model_list()
        model_list.remove(self.model_layers[-1][0])
        model_to_remove = random.choice(model_list)
        self.remove_model(model_to_remove)
        self.input_layer.remove(model_to_remove.output_node)


    def add_model(self):
        layer_to_create_model_on = random.randint(0, self.max_layers - 2)
        valid_inputs = self.get_valid_inputs_for_layer(layer_to_create_model_on)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
        model_output = Node(layer_to_create_model_on + 1, self.get_next_n_id(layer_to_create_model_on + 1))
        self.input_layer.append(model_output)
        self.model_layers[layer_to_create_model_on].append(
            self.get_random_model(layer_to_create_model_on, input_nodes=model_inputs, output_node=model_output))


    def reset_model(self):
        model_list = self.get_model_list()
        model_list.remove(self.model_layers[-1][0])
        changing_model = random.choice(model_list)
        self.change_model(changing_model)
        layer_to_create_model_on = changing_model.l_id
        valid_inputs = self.get_valid_inputs_for_layer(layer_to_create_model_on)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
        changing_model.reset_model(model_inputs)
        changing_model.set_random_hyparameters()


    def train(self, x, y, n_folds = 'auto'):
        '''
        Given data and labels this splits it into folds and iteratively trains each layer
        on the outputs of previous models which predict a fold it has not trained on.
        '''

        #print(self.gen, len(self.input_layer))

        if n_folds == 'auto':
            split_num = self.max_layers
            xs = np.array_split(x, split_num)
            ys = np.array_split(y, split_num)
        elif isinstance(n_folds, int):
            split_num = n_folds
            xs = np.array_split(x, split_num)
            ys = np.array_split(y, split_num)
        else:
            raise AttributeError('n_folds must be an int or "auto"')


        for l_count, l in enumerate(self.model_layers):
            train_x = xs[l_count%split_num]
            train_y = ys[l_count%split_num]

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
                m.score(x, y)

        self.last_tested_score = self.model_layers[-1][0].score(x, y)
        return self.last_tested_score

    def predict(self, x):
        self.load_input_layer(x)
        for l in self.model_layers:
            for m in l:
                m.predict(x, None)

        return self.model_layers[-1][0].output_node.dump()


    def remove_model(self, m):
        model_output_node = m.output_node

        for i in self.model_layers:
            for j in i:
                if model_output_node in j.input_nodes and len(j.input_nodes) > 1:
                    j.input_nodes.remove(model_output_node)
                elif model_output_node in j.input_nodes:
                    j.input_nodes.remove(model_output_node)
                    layer_to_create_model_on = m.l_id
                    valid_inputs = self.get_valid_inputs_for_layer(layer_to_create_model_on)
                    if j in valid_inputs:
                        valid_inputs.remove(j)
                    model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
                    j.input_nodes = model_inputs
        del model_output_node
        self.model_layers[m.l_id].remove(m)


    def change_model(self, m):
        model_output_node = m.output_node
        model_layer = m.l_id
        model_index = self.model_layers[model_layer].index(m)

        self.model_layers[model_layer][model_index] = self.get_random_model(model_layer, m.input_nodes, model_output_node)


    def get_model_count(self):
        return sum([len(i) for i in self.model_layers])


    def get_model_list(self):
        model_list = []
        for i in self.model_layers:
            model_list.extend(i)
        return model_list

    def select_random_model(self):
        return random.choice(self.get_model_list())



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
        nodes = []

        for i in range(l+1):
            nodes.extend(self.get_input_layer_at_layer_l(i))
        return nodes
        # if l == 0 or l == 1:
        #     return self.get_input_layer_at_layer_l(0)
        # else:
        #     return self.get_input_layer_at_layer_l(0) + self.get_input_layer_at_layer_l(l - 1)


    def generate_first_model(self):
        '''
        creates single model at the last layer
        '''
        valid_inputs = self.get_valid_inputs_for_layer(self.max_layers)
        model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))

        self.model_layers[-1].append(self.get_random_model(len(self.model_layers) - 1, model_inputs, self.output_node))


    def generate_random_stack(self):
        '''
        creates a random set of models
        '''

        models_per_layer = self.max_models//self.max_layers
        for l in range(len(self.model_layers)):

            valid_inputs = self.get_valid_inputs_for_layer(l)
            for m in range(models_per_layer):
                model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
                model_output = Node(l + 1,
                                    self.get_next_n_id(l))
                self.model_layers[l].append(self.get_random_model(l, model_inputs, model_output))


    def get_next_n_id(self, l):
        if len([i.n_id for i in self.input_layer if i.l_id == l]) > 0:
            return max([i.n_id for i in self.input_layer if i.l_id == l]) + 1
        else:
            return 0


    #TODO: get random model from allowed set
    def get_random_model(self, l_id, input_nodes=None, output_node=None):
        model_index = random.randint(0, num_of_models -1)

        possible_models = [
                            'AdaBoostClassifierModel',
                           'RandomForestClassifierModel',
                           'ExtraTreesClassifierModel',
                           'GradientBoostingClassifierModel',
                           'DecisionTreeClassifierModel',
                           'XGBoosterModel',
                           #'CatboostRegressorModel',
                           'LGBMRegressorModel'
                           ]

        return eval(random.choice(possible_models))(l_id, input_nodes=input_nodes, output_node=output_node)

        # if model_index == 0:
        #     return AdaBoostClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        # if model_index == 1:
        #     return RandomForestClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        # if model_index == 2:
        #     return ExtraTreesClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        # if model_index == 3:
        #     return GradientBoostingClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        # if model_index == 4:
        #     return DecisionTreeClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)
        # # if model_index == 5:
        # #     return KNeighborsClassifierModel(l_id, input_nodes=input_nodes, output_node=output_node)


    def generate_dot_file(self):
        g = Digraph('G {0}'.format(self.gen), filename='process.gv', engine='sfdp')

        for i in self.get_model_list():
            for n in i.input_nodes:

                g.edge('node {0} {1}'.format(n.l_id, n.n_id), 'model {0} {1} {2} {3}'.format(i.__class__.__name__, i.l_id, id(i), i.last_tested_score))
            g.edge( 'model {0} {1} {2} {3}'.format(i.__class__.__name__, i.l_id, id(i), i.last_tested_score), 'node {0} {1}'.format(i.output_node.l_id, i.output_node.n_id))
        g.save('process2.gv')


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
        self.last_tested_score = 0


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
            if len(j) == 1:
                selected_hyper_parameters[i] = j[0]
            elif isinstance(j[0], int) and len(j) == 2:
                selected_hyper_parameters[i] = random.randint(j[0], j[1])
            elif isinstance(j[0], float):
                selected_hyper_parameters[i] = random.uniform(j[0],j[-1])
            else:
                selected_hyper_parameters[i] = random.choice(j)
        return selected_hyper_parameters


    def reset_model(self, input_nodes = None):
        self.input_nodes = input_nodes


    def get_input(self):
        output_list = [i.dump() for i in self.input_nodes]
        # for i in output_list:
        #     print(i.shape)
        # print()
        output_array = np.vstack([i for i in output_list])
        return output_array.T

    def fit(self, x, y):
        input_x = self.get_input()
        self.clf.fit(input_x, y)

    def predict(self, x, y):
        input_x = self.get_input()
        self.output_node.load(self.clf.predict_proba(input_x)[:,1])

    def score(self, x, y):
        input_x = self.get_input()
        self.last_tested_score = self.clf.score(input_x, y)
        return self.last_tested_score

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

class XGBoosterModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {
                                    'learning_rate': (0.001, 1.0),
                                    'min_child_weight': (0, 10),
                                    'max_depth': (0, 50),
                                    'max_delta_step': (0, 20),
                                    'subsample': (0.01, 1.0),
                                    'colsample_bytree': (0.01, 1.0),
                                    'colsample_bylevel': (0.01, 1.0),
                                    'reg_lambda': (1e-9, 1000),
                                    'reg_alpha': (1e-9, 1.0),
                                    'gamma': (1e-9, 0.5),
                                    'n_estimators': (50, 200)
                                }
        self.set_hyperparameters = dict()
        #self.clf = xgb.Booster(params=self.get_random_hyperparameters())

    '''loading happens at fitting'''
    def load_model(self):
        pass
        #self.clf =  xgb.Booster(**self.set_hyperparameters)


    def predict(self, x, y):
        input_x = self.get_input()
        dtest = xgb.DMatrix(input_x)
        self.output_node.load(self.clf.predict(dtest))
        return self.output_node.dump()


    def score(self, x, y):
        input_x = self.get_input()
        preds = self.predict(input_x, y)
        prediction = np.rint(preds)

        self.last_tested_score = accuracy_score(prediction, y)
        return self.last_tested_score


    def fit(self, x, y):
        input_x = self.get_input()
        dtrain = xgb.DMatrix(input_x, y)
        self.clf = xgb.train(self.set_hyperparameters, dtrain)


class CatboostRegressorModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {'iterations':[100, 500],
                                  'learning_rate': [.01, .2],
                                  'depth': [2,12]
                                  }
        self.set_hyperparameters = dict()
        self.clf = catboost.CatBoostRegressor(**self.get_random_hyperparameters())

    def load_model(self):
        self.clf =  catboost.CatBoostRegressor(**self.set_hyperparameters)


    def score(self, x, y):
        input_x = self.get_input()
        prediction = np.rint(self.clf.predict(input_x, y))

        self.last_tested_score = accuracy_score(prediction, y)
        return self.last_tested_score




class LGBMRegressorModel(Model):

    def __init__(self, l_id, input_nodes = None, output_node = None):
        super().__init__( l_id, input_nodes = input_nodes, output_node = output_node)
        self.hyperparam_ranges = {
                                    'learning_rate': (0.001, 1.0),
                                    'num_leaves': (2, 300),
                                    'max_depth': (2, 50),
                                    'min_data_in_leaf': (1, 50),
                                    'max_bin': (100, 1000),
                                    'subsample': (0.01, 1.0),
                                    'subsample_freq': (0, 10),
                                    'colsample_bytree': (0.01, 1.0),
                                    'min_child_weight': (1, 10),
                                    'subsample_for_bin': (100000, 500000),
                                    'reg_lambda': (1e-9, 1000),
                                    'reg_alpha': (1e-9, 1.0),
                                    'n_estimators': (50, 150),
                                }
        self.set_hyperparameters = dict()
        #self.clf = lgb.Booster(params = self.get_random_hyperparameters())

    '''loading happens at fitting'''
    def load_model(self):
        pass
        #self.clf = lgb.Booster(params = self.set_hyperparameters)

    def fit(self, x, y):
        input_x = self.get_input()
        dtrain = lgb.Dataset(input_x, label=y)
        self.clf = lgb.train(params = self.set_hyperparameters, train_set=dtrain)


    def predict(self, x, y):
        input_x = self.get_input()
        dtest = lgb.Dataset(input_x)
        self.output_node.load(self.clf.predict(input_x))
        return self.output_node.dump()


    def score(self, x, y):
        input_x = self.get_input()
        preds = self.predict(input_x, y)
        prediction = np.rint(preds)

        self.last_tested_score = accuracy_score(prediction, y)
        return self.last_tested_score


class Node():

    def __init__(self, l_id, n_id):
        self.n_id = n_id
        self.l_id = l_id
        self.value = None

    def load(self, x):
        self.value = x

    def dump(self):
        return self.value


def test_income_dataset():
    df = pd.read_csv('income.csv', nrows = 10000,
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


def test_graph():
    with open('model_stacking_save_dir/483.plk', 'rb') as infile:
        stack = pickle.load(infile)

    stack.generate_dot_file()

def compare_best_model():
    df = pd.read_csv('income.csv',
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
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)

    with open('model_stacking_save_dir/73.plk', 'rb') as infile:
        stack = pickle.load(infile)

    stack.load_models()
    stack.train(train_x, train_y)
    stack.score(test_x, test_y)
    stack.del_models()

    print('stack score:', stack.last_tested_score)

    clf = ensemble.RandomForestClassifier(n_estimators=100)
    clf.fit(train_x, train_y)
    single_rf_score = clf.score(test_x, test_y)
    print('single_rf_score:',single_rf_score)

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(train_x, train_y)
    single_gbm_score = clf.score(test_x, test_y)
    print('single_gbm_score:', single_gbm_score)



def compare_best_model2():
    df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\fraud\proccessed_train2.csv', sep = '|')
    print(df.columns)
    df = df.sample(n = 10000000)
    df_pos = df.loc[df['is_attributed'] == 1]
    df_neg = df.loc[df['is_attributed'] == 0]
    df_neg = df_neg.sample(n = df_pos.shape[0])
    del df
    gc.collect()
    df = pd.concat([df_pos, df_neg])
    df = df.sample(frac = 1)

    print(df.shape)

    y = df['is_attributed']
    x = df.drop(['is_attributed', 'attributed_time'], axis=1)
    x = x.as_matrix()
    y = y.as_matrix()
    Builder(x=x, y=y)

    gc.collect()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.2)
    with open('model_stacking_save_dir/50.plk', 'rb') as infile:
        stack = pickle.load(infile)

    stack.load_models()
    stack.train(train_x, train_y)
    stack.score(test_x, test_y)
    stack.del_models()

    print('stack score:', stack.last_tested_score)

    clf = ensemble.RandomForestClassifier(n_estimators=100)
    clf.fit(train_x, train_y)
    single_rf_score = clf.score(test_x, test_y)
    print('single_rf_score:',single_rf_score)

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(train_x, train_y)
    single_gbm_score = clf.score(test_x, test_y)
    print('single_gbm_score:', single_gbm_score)




def extract_features_from_row(row_tuple):
    _, row = row_tuple

    pj_class = [0 for _ in range(3)]
    pj_class[row['Pclass']-1] = 1

    name = row['Name']
    name_part_array = [0 for i in range(5)]
    if 'mr.' in name.lower():
        name_part_array[0] = 1
    elif 'mrs.' in name.lower():
        name_part_array[1] = 1
    elif 'miss' in name.lower():
        name_part_array[2] = 1
    else:
        name_part_array[3] = 1
    if '(' in name and ')' in name:
        name_part_array[4] = 1
    if row['Sex'] == 'male':
        sex = 1
    else:
        sex = 0
    sibsp = row['SibSp']
    parch = row['Parch']
    fare = row['Fare']
    name = row['Name']
    age = row['Age']
    is_age_there = int(row['is_age_there'])
    is_cabin_there = int(row['is_cabin_there'])

    embarked_part_array = [0 for i in range(4)]
    if 'S' in name.lower():
        embarked_part_array[0] = 1
    elif 'C' in name.lower():
        embarked_part_array[1] = 1
    elif 'Q' in name.lower():
        embarked_part_array[2] = 1
    else:
        embarked_part_array[3] = 1

    output_x = [sex, sibsp, parch, fare, age, is_age_there,
                is_cabin_there] + embarked_part_array + name_part_array + pj_class
    return output_x


def get_features(input_df):
    output = dict()

    input_df['is_age_there'] = pd.notnull(input_df['Age'])
    input_df['is_cabin_there'] = pd.notnull(input_df['Cabin'])
    input_df = input_df.fillna(input_df.median())
    max_list = []

    for i in input_df.iterrows():
        output_x = extract_features_from_row(i)
        if len(max_list) == 0:
            max_list = [0 for _ in output_x]

        for j in range(len(output_x)):
            max_list[j] = max(max_list[j], output_x[j])

        #output_x = [sex] + embarked_part_array
        try:
            output_y = [i[1]['Survived']]
        except:
            output_y = None
        output[i[1]['PassengerId']] = [output_x, output_y]


    return pd.DataFrame.from_dict(output)


def test_titanic():
    train_df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\titanic\train.csv')

    train_df = get_features(train_df).T

    x = np.vstack(train_df[0])
    y = np.vstack(train_df[1])
    y = np.ravel(y)
    Builder(x=x, y=y)

    test_df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\titanic\test.csv')
    test_df2 = get_features(test_df).T
    test_x = np.vstack(test_df2[0])
    test_y = np.vstack(test_df2[1])

    train_df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\titanic\train.csv')
    train_df = get_features(train_df).T
    train_x = np.vstack(train_df[0])
    train_y = np.vstack(train_df[1])

    with open('model_stacking_save_dir/138.plk', 'rb') as infile:
        stack = pickle.load(infile)

    stack.load_models()
    stack.train(train_x, train_y)

    test_df['Survived'] = np.rint(stack.predict(test_x))
    test_df = test_df[['PassengerId', 'Survived']]
    test_df['Survived'] = test_df['Survived'].astype(int)
    test_df['Survived'] = test_df['Survived'].apply(lambda x: 1 if x == 0 else 0)
    test_df.to_csv('titanic_preds.csv', index = False)

    stack.del_models()


def preproccess(x, models):

    if not models:
        models = []
        training_set = x.as_matrix()
        training_set2 = x.as_matrix()
        np.random.shuffle(training_set)
        x_splits = np.array_split(training_set, 100)

        for count, i in enumerate(x_splits):
            cluster_num = random.randint(10, 25)
            model = MiniBatchKMeans(n_clusters=cluster_num)
            model.fit(training_set)
            models.append(model)
            print('training', count)

        for count, m in enumerate(models):
            x[count] = m.predict(training_set2)
            print('predicting', count)

    else:
        training_set = x.as_matrix()
        for count, m in enumerate(models):
            x[count] = m.predict(training_set)

    return x.as_matrix(), models


def test_numerai(generations = 200):

    path = r'C:\Users\tdelforge\Documents\Kaggle_datasets\numerai\numerai_datasets/'
    training_data = pd.read_csv(path + 'numerai_training_data.csv', header=0)
    prediction_data = pd.read_csv(path + 'numerai_tournament_data.csv', header=0)
    print('data read')

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]

    x = training_data[features]
    x, cluster_models = preproccess(x, None)
    y = training_data["target"].as_matrix()

    x_design, _, y_design, _ = train_test_split(x, y, train_size=.3)

    Builder(x=x, y=y, max_generations=generations)
    with open('model_stacking_save_dir/{0}.plk'.format(generations), 'rb') as infile:
        stack = pickle.load(infile)

    x_prediction = prediction_data[features]
    x_prediction, cluster_models = preproccess(x_prediction, cluster_models)
    ids = prediction_data["id"].to_frame()

    stack.load_models()
    stack.train(x, y)
    ids['probability'] = stack.predict(x_prediction)
    stack.del_models()
    ids.to_csv("predictions.csv", index=False)



if __name__ == '__main__':
    # test_income_dataset()
    #test_graph()
    #compare_best_model()
    #test_titanic()
    # compare_best_model2()
    test_numerai()

