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
                # TODO: Code for continuous attributes
                print('cont')
            else:
                unique_vals = data[col].unique()
                for val in unique_vals:
                    rules.append(Rule(col, val, is_cat=True))
                    true_rows = data.loc[data[col] == val]
                    weighted_entropy += ((true_rows.shape[0] / data.shape[0]) *
                                         utils.entropy(true_rows.iloc[:, -1]))
            gain = init_entropy - weighted_entropy
            if gain > max_gain and math.fabs(gain) > 1e-3:
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
        if gain < 1e-3 or majority_class / data.shape[0] > 0.95:
            return Node(majority=majority_class, is_leaf=True)

        children = []
        if max_col in self.cont_cols:
            # TODO: Code for continuous attributes
            print('cont')
        else:
            for rule in rules:
                val = rule.value
                children.append(self.build_tree(data.loc[data[max_col] == val],
                                                depth+1))
        return Node(children, rules, majority_class, False)

    def fit(
        self,
        X_train,
        Y_train
    ):
        data = X_train.join(Y_train)
        self.root = self.build_tree(data, 0)

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
        is_leaf=False
    ):
        self.children = children
        self.rules = rules
        self.majority = majority
        self.is_leaf=is_leaf


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