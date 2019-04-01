import numpy as np
import math
import utils

class DTree:
    
    def __init__(
        self,
        cont_cols
    ):  
        self.attr_split_count = {}
        self.cont_cols = set(cont_cols)
        self.root = None
        print('init')
    
    
    def best_split(
        self,
        data
    ):
        init_entropy = utils.entropy(data['Y'])
        num_rows = data.shape[0]

        col_list = data.columns.to_list()[:-1]

        max_gain = 0
        max_rules = None

        for col in col_list:
            gain = 0
            weighted_entropy = 0
            rules = []
            if col in self.cont_cols:
                # Code for continuous attributes
                print('cont')
            else:
                unique_vals = data[col].unique()
                for val in unique_vals:
                    rules.append(Rule(col, val, is_cat=True))
                    true_rows = data.loc[data[col] == val]
                    weighted_entropy += ((true_rows.shape[0] / data.shape[0]) *
                                          utils.entropy(true_rows.iloc[:, -1]))
            
            gain = init_entropy - weighted_entropy
            if gain > max_gain:
                max_gain = gain
                max_rules = rules

        return max_gain, max_rules        

    def build_tree(
        self,
        data
    ):
        # Find best split feature
        # Loop over all possible values of feature to create subtrees
        print('b')
        max_gain, rules = self.best_split(data)
        print(rules[0].col, max_gain)
        return Node()

    def fit(
        self,
        X_train,
        Y_train
    ):
        data = X_train.join(Y_train)
        self.root = self.build_tree(data)

class Node:

    def __init__(
        self
    ):
        self.children = []
        self.rules = []


class Rule():

    def __init__(
        self,
        col,
        value,
        is_cat
    ):
        self.col = col
        self.value = value
        self.is_cat = is_cat

    def compare(
        self,
        value 
    ):
        if self.is_cat:
            return value == self.value
        else:
            return value >= self.value
