from sklearn import datasets
import pandas as pd
import numpy as np
import random as rand
import math


def get_feature_splits(y, f):

    '''split the y variable by split s in feature f'''

    return ([{'feature' : f.name, 'split' :  s, 'left': y[f < s],
              'right' :  y[f >= s]} for s in f])


def combined_RSS(y1, y2):

    '''calculate the combined residual sum of squares for y1 and y1'''

    return np.sum((y1 - np.mean(y1))**2) + np.sum((y2 - np.mean(y2))**2)


def cost_function(feature_split):

    '''calculate the total RSS for the splits defined by splitting feature f at split s'''

    return ({'feature' : feature_split['feature'],
              'split' : feature_split['split'],
              'cost' : combined_RSS(feature_split['left'], feature_split['right'])
            }
           )


def flatten_list(l):
    '''flattens list of lists to list'''
    return [item for sublist in l for item in sublist]


def get_split_costs(df):

    '''calulate the cost for each split'''

    return flatten_list(
        [[cost_function(split) for split in get_feature_splits(
            df.iloc[:,-1], df.iloc[:,:-1].loc[:, feature])] for feature in rand.sample(list(
            df.iloc[:,:-1].columns), int(math.sqrt(len(list(df.iloc[:,:-1].columns)))))])


def select_node(split_costs):

    '''select the lowest cost node'''

    return (rand.choice([{'feature' : split['feature'], 'split' : split['split'],
                          'cost' : split['cost']}
                         for split in split_costs if split['cost'] == np.min(
                             [split['cost'] for split in split_costs])]))


def terminal_node(node_y):

    '''returns the prediction for the terminal node'''

    return np.mean(node_y)


def grow_tree(df, node, depth, max_depth = 1, min_size = 5 ):

    ''' recursively grows a decision tree by applying the function to each node it
    returns until max depth or min size criteria is met '''

    left = df.loc[df.loc[:, node['feature']] < node['split']]
    right = df.loc[df.loc[:, node['feature']] >= node['split']]

    if left.empty or right.empty:
        return terminal_node(list(left.iloc[:, -1]) + list(right.iloc[:, -1]))

    elif depth >= max_depth:
        return {'node': node,
                'left': terminal_node(left.iloc[:, -1]),
                'right': terminal_node(right.iloc[:, -1])}

    else:
        return {'node' : node,

                'left' : (lambda x: terminal_node(list(x.iloc[:, -1]))
                          if len(x.iloc[:, -1]) <= min_size
                          else grow_tree(x, select_node(get_split_costs(x)),
                                        depth + 1, max_depth, min_size))(left),

                'right' : (lambda x: terminal_node(list(x.iloc[:, -1]))
                           if len(x.iloc[:, -1]) <= min_size
                           else grow_tree(x, select_node(get_split_costs(x)),
                                        depth + 1, max_depth, min_size))(right)
               }


def bootstrap(df, random_state = 1):

    ''' generates a bootstrap sample, can be parameterised by random_state'''

    return df.sample(len(df), replace = True, random_state = random_state)


def grow_random_forest(df, max_depth = 1, min_size = 5, no_of_trees = 10):

    ''' grows a random forest by growing a tree on no_of_trees bootstrap samples
    of df '''

    return [grow_tree(
        bootstrap(df, i), select_node( get_split_costs(bootstrap(df, i))),
        1, max_depth, min_size) for i in range(0, no_of_trees)]


def tree_predict_row(row, tree):

    '''uses the trained tree to partition the feature space of the test observation and return
    the partition prediction value, which for regression is the mean of the partition'''

    if row[tree['node']['feature']] <  tree['node']['split']:
        if not isinstance(tree['left'], dict):
            return tree['left']
        else:
            return tree_predict_row(row, tree['left'])
    else:
         if not isinstance(tree['right'], dict):
            return tree['right']
         else:
            return tree_predict_row(row, tree['right'])


def forest_predict_row(row, forest):

    ''' predicts each row of df for each tree than takes the average across
    all trees for each row '''

    return np.mean([tree_predict_row(row, tree) for tree in forest])


def predict(df, forest):

    ''' applies predict row function to a df of test observations '''

    return df.apply(forest_predict_row, axis = 1, forest = forest)


def mse(y, predicted_y):

    ''' returns the mean square error: the sum of squyared differences between predicted y and actual y'''

    return np.sum((y - predicted_y)**2)


def total_sum_of_squares(y):

    '''returns the total sum of squares: the sum of squared difference between y and the mean of y i.e var(y)'''

    return np.sum((y - np.mean(y))**2)


def r_squared(y, predicted_y):

    ''' returns to proportion of variance attributable the model: 1 - the ratio of mse to total sum of squares.'''

    return 1 - (mse(y, predicted_y)/total_sum_of_squares(y))
