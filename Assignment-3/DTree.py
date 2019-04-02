import math
import utils
import pickle
from pathlib import Path


class DTree:

    def __init__(
        self,
        cont_cols,
        gain_threshold=1e-3,
        purity_threshold=0.95
    ):
        self.attr_split_count = {}
        self.cont_cols = set(cont_cols)
        self.root = None
        self.gain_th = gain_threshold
        self.purity_th = purity_threshold

    def best_split(
        self,
        data
    ):
        init_entropy = utils.entropy(data['Y'])
        # num_rows = data.shape[0]

        col_list = data.columns.to_list()[:-1]

        max_gain = 0
        max_rules = None
        max_col = None

        for col in col_list:
            gain = 0
            weighted_entropy = 0
            rules = []
            if col in self.cont_cols:
                median = data[col].median()
                rules.append(Rule(col, median, is_cat=False))

                true_rows = data.loc[data[col] >= median]
                false_rows = data.loc[data[col] < median]

                weighted_entropy += (
                        (true_rows.shape[0] / data.shape[0] *
                         utils.entropy(true_rows.iloc[:, -1])) +
                        (false_rows.shape[0] / data.shape[0] *
                         utils.entropy(false_rows.iloc[:, -1])))
            else:
                unique_vals = data[col].unique()
                for val in unique_vals:
                    rules.append(Rule(col, val, is_cat=True))
                    true_rows = data.loc[data[col] == val]
                    weighted_entropy += ((true_rows.shape[0] / data.shape[0]) *
                                         utils.entropy(true_rows.iloc[:, -1]))
            gain = init_entropy - weighted_entropy
            if gain > max_gain and math.fabs(gain) > self.gain_th:
                max_gain = gain
                max_rules = rules
                max_col = col

        return max_gain, max_rules, max_col

    def build_tree(
        self,
        data,
        depth
    ):
        # Loop over all possible values of feature to create subtrees
        gain, rules, max_col = self.best_split(data)

        majority_class = data.iloc[:, -1].value_counts().idxmax()
        if (gain < self.gain_th or
           (majority_class / data.shape[0]) > self.purity_th):
            return Node(majority=majority_class, is_leaf=True, depth=depth)

        children = []
        if max_col in self.cont_cols:
            # TODO: Code for continuous attributes
            print('cont')
        else:
            for rule in rules:
                val = rule.value
                children.append(self.build_tree(data.loc[data[max_col] == val],
                                                depth+1))
        return Node(children, rules, majority_class, False, depth=depth)

    def fit(
        self,
        X_train,
        Y_train
    ):
        pickle_file = Path('pickle_a')
        if pickle_file.is_file():
            self.root = pickle.load(open(pickle_file, 'rb'))
        else:
            data = X_train.join(Y_train)
            self.root = self.build_tree(data, 0)
            pickle.dump(self.root, open(pickle_file, 'wb'))

    def predict(
        self,
        X_test
    ):
        Y_pred = []
        for x in X_test.itertuples():
            Y_pred.append(self.model(x))
        return Y_pred

    def model(
        self,
        x
    ):
        classified = False
        curr_node = self.root
        while ((not classified) and (not curr_node.is_leaf)):
            curr_is_cat = curr_node.rules[0].is_cat
            curr_col = curr_node.rules[0].col
            can_go_further = False
            if curr_is_cat:
                for rule, idx in zip(curr_node.rules,
                                     range(len(curr_node.rules))):
                    if getattr(x, curr_col) == rule.value:
                        curr_node = curr_node.children[idx]
                        can_go_further = True
                        continue
            else:
                # TODO
                print('continuous')

            if not can_go_further:
                classified = True

        return curr_node.majority


class Node:

    def __init__(
        self,
        children=[],
        rules=[],
        majority=None,
        is_leaf=False,
        depth=0
    ):
        self.children = children
        self.rules = rules
        self.majority = majority
        self.is_leaf = is_leaf
        self.depth = depth


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
