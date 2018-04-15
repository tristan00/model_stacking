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
from graphviz import Graph, Digraph


num_of_models = 5


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory


class Builder():
    def __init__(self, max_num_of_stacks = 10, mutations_per_model = 2, x=None, y = None, serialize_dir = '', max_generations = 100):
        stacks = [Stack(gen=0, x=x, y=y) for _ in range(max_num_of_stacks)]
        save_dir = create_directory(serialize_dir + 'model_stacking_save_dir')
        score_dicts = []

        gen_count = 0
        for _ in range(max_generations):
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
    def __init__(self, max_layers = 3, max_models = 12, x = None, y = None, gen = 0):
        self.gen = 0
        self.max_layers = max_layers
        self.max_models = max_models
        #self.output_model = None
        self.model_layers = [[] for _ in range(max_layers)]
        self.min_input_layer_choice = 1
        self.input_layer = [Node(0, i) for i in range(x.shape[1])]
        self.output_node = Node(max_layers + 1, 0)
        self.generate_first_model()
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


        if mutation_level == 'random':
            light_mutation = random.randint(0,1)
        else:
            light_mutation =0

        if light_mutation == 1 and len(stack_copy.get_model_list()) > 1:
            add_edge = random.randint(0, 1)
            remove_edge = random.randint(0, 1)

            if add_edge == 1:
                models = stack_copy.get_model_list()
                models.remove(stack_copy.model_layers[-1][0])
                model_to_modify = random.choice(models)
                valid_inputs = stack_copy.get_valid_inputs_for_layer(model_to_modify.l_id)
                valid_inputs = [i for i in valid_inputs if valid_inputs not in model_to_modify.input_nodes]
                if len(valid_inputs)>0:
                    model_to_modify.input_nodes.append(random.choice(valid_inputs))
            if remove_edge == 1:
                models = stack_copy.get_model_list()
                models.remove(stack_copy.model_layers[-1][0])
                model_to_modify = random.choice(models)

                if len(model_to_modify.input_nodes)>2:
                    model_to_modify.input_nodes.remove(random.choice(model_to_modify.input_nodes))

        else:
            reset_model = random.randint(0, 1)
            # create_model = random.randint(0, 1)
            del_model = random.randint(0, 1)

            '''
            remove model if valid to do so
            '''
            if stack_copy.get_model_count() >= stack_copy.max_models and del_model == 1:
                model_list = stack_copy.get_model_list()
                model_list.remove(stack_copy.model_layers[-1][0])
                model_to_remove = random.choice(model_list)
                stack_copy.remove_model(model_to_remove)
                stack_copy.input_layer.remove(model_to_remove.output_node)


            '''
            create model if valid to do so
            '''
            if stack_copy.get_model_count() < stack_copy.max_models:
                layer_to_create_model_on = random.randint(0, stack_copy.max_layers - 2)
                valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
                model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
                model_output = Node(layer_to_create_model_on + 1, stack_copy.get_next_n_id(layer_to_create_model_on + 1))
                stack_copy.input_layer.append(model_output)
                stack_copy.model_layers[layer_to_create_model_on].append(stack_copy.get_random_model(layer_to_create_model_on, input_nodes=model_inputs, output_node=model_output))



            '''
            reset random model
            '''
            if reset_model == 1 and len(stack_copy.get_model_list()) > 1:
                model_list = stack_copy.get_model_list()
                model_list.remove(stack_copy.model_layers[-1][0])
                changing_model = random.choice(model_list)
                stack_copy.change_model(changing_model)
                layer_to_create_model_on = changing_model.l_id
                valid_inputs = stack_copy.get_valid_inputs_for_layer(layer_to_create_model_on)
                model_inputs = random.sample(valid_inputs, random.randint(2, len(valid_inputs)))
                changing_model.reset_model(model_inputs)
                changing_model.set_random_hyparameters()


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
            if isinstance(j[0], int) and len(j) == 2:
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
    with open('model_stacking_save_dir/111.plk', 'rb') as infile:
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

    with open('model_stacking_save_dir/111.plk', 'rb') as infile:
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


def rank_df(df, columns, name):
    print('rank', columns)
    df2 = df.groupby(columns).count().add_suffix('_count').reset_index()
    df2 = df2[columns + ['counting_column_count']]
    df2[name] = df2['counting_column_count'].rank(method='dense')
    max_rank = max(df2[name])
    df2[name] /= max_rank
    df2 = df2[columns + [name]]
    df2[name] = df2[name].astype('float32')
    df = df.merge(df2, on=columns)
    return df

def count_df(df, columns, name):
    print('count', columns)
    df[name] = df.groupby(columns).cumcount()
    df[name] = df[name].astype('uint16')
    return df

def time_since_last(df, columns, name):
    print('time_since_last', columns)
    df[name] = df.groupby(columns)['click_time'].diff()
    df[name] = df[name].fillna(-1)
    df[name] = df[name].astype(int)
    mean_std = df.groupby(columns)[name].agg([np.median, np.std]).reset_index()
    df = df.merge(mean_std, on = columns)
    df[name] = df[name].fillna(-1)
    df[name + '_std'] = df['std'].fillna(-1)
    df[name + '_median'] = df['median'].fillna(-1)
    df = df.drop(['median', 'std'], axis = 1)
    return df

def time_till_next(df, columns, name):
    print('time_till_next', columns)
    df[name] = df.groupby(columns)['click_time'].transform(
        lambda x: x.diff().shift(-1))
    df[name] = df[name].fillna(-1)
    df[name] = df[name].astype(int)
    mean_std = df.groupby(columns)[name].agg([np.median, np.std]).reset_index()
    df = df.merge(mean_std, on = columns)
    df[name] = df[name].fillna(-1)
    df[name + '_std'] = df['std'].fillna(-1)
    df[name + '_median'] = df['median'].fillna(-1)
    df = df.drop(['median', 'std'], axis = 1)
    return df

def preproccess_df(df):
    print(df.shape, df.columns)
    df['counting_column'] = 1

    df['datetime'] = df['click_time'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    df['click_hour'] = df['datetime'].apply(lambda x: x.hour).astype('uint8')
    df['click_day'] = df['datetime'].apply(lambda x: x.day).astype('uint8')
    df['click_second'] = df['datetime'].apply(lambda x: x.second).astype('uint8')
    df['click_minute'] = df['datetime'].apply(lambda x: x.minute).astype('uint8')
    #df['click_minute'] /= 60 #TODO: change to 60 on new models
    df['click_time'] = df['datetime'].apply(lambda x: x.timestamp())
    df = df.drop(['datetime'], axis = 1)

    df = df.sort_values(by=['click_time'])

    df['click_hour'] = df['click_hour'].apply(lambda x: 5 if x == 6 else 10 if x == 11 else 14)


    print('time added', df.shape)
    #


    possible_names = ['ip', 'device', 'os', 'channel', 'click_day']

    # for l in range(len(possible_names)):
    #     combinations = itertools.combinations(possible_names, l+1)
    #     for i in combinations:
    #         if 'click_day' not in i or len(i) == 1 or len(i) == 5:
    #             continue
    #         print(i, time.time() - start_time)
    #         df = time_since_last(df, list(i), '_'.join(i)+'_next')
    #         gc.collect()
    #         # df = time_till_next(df, list(i), '_'.join(i) + '_last')
    #         # gc.collect()


    rank_list = [['ip', 'os', 'device', 'channel', 'click_day'],
                 ['ip', 'device', 'os', 'click_day']]
    for i in rank_list:
        df = time_since_last(df, list(i), '_'.join(i) + '_last')
        gc.collect()
        df = time_till_next(df, list(i), '_'.join(i) + '_next')
        gc.collect()
        df = count_df(df, list(i), '_'.join(i) + '_count')
        gc.collect()


    rank_list = [['ip', 'click_day'],
                 ['ip', 'device', 'click_day'],
                 ['ip', 'os', 'click_day'],
                 ['device', 'os', 'click_day'],
                 ['ip', 'device', 'channel', 'click_day'],
                 ['channel', 'click_day'],
                 ['ip', 'device', 'os', 'click_day'],
                 ['ip', 'channel', 'click_day'],
                 ['device', 'os', 'channel', 'click_day'],
                 ['os', 'channel', 'click_day'],
                 ['device', 'click_day'],
                 ['device', 'channel', 'click_day'],
                 ['os', 'click_day'],
                 ['app', 'channel', 'click_day']]
    for i in rank_list:
        df = rank_df(df, list(i), '_'.join(i) + '_rank')
        df = df.sort_values(by=['click_time'])
        gc.collect()

    # count_lists = [
    #             ['ip', 'device', 'os', 'channel', 'app'],
    #             ['click_hour', 'click_day', 'ip', 'device', 'os'],
    #                ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel'],
    #                ['click_hour', 'click_day', 'ip', 'device', 'os', 'channel', 'app']]
    #
    # print(time.time() - start_time)
    # for i in count_lists:
    #     df = rank_df(df, list(i), '_'.join(i) + '_rank')

    df['device'] = df['device'].astype('int')
    df['os'] = df['os'].astype('int')
    df['channel'] = df['channel'].astype('int')
    df['click_hour'] = df['click_hour'].astype('int')
    df = df.drop(['ip','click_time', 'counting_column', 'click_day', 'click_day'], axis=1)


    print(df.shape)


    #df.drop(['counting_column'], axis=1, inplace=True)
    return df


def compare_best_model2():
    df = pd.read_csv(r'C:\Users\tdelforge\Documents\Kaggle_datasets\fraud\proccessed_train2.csv', sep = '|')
    print(df.columns)
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

    path = r'C:/Users/tdelforge/Documents/Kaggle_datasets/fraud/'
    test1 = pd.read_csv(path + "test.csv")

    sub = pd.DataFrame()
    sub['click_id'] = test1['click_id']

    test1.drop('click_id', axis=1, inplace=True)
    test1 = preproccess_df(test1)


    stack.load_models()
    stack.train(x, y)
    stack.predict()
    stack.del_models()



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



if __name__ == '__main__':
    #test_income_dataset()
    #test_graph()
    #compare_best_model()
    #test_titanic()
    compare_best_model2()

