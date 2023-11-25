"""
Build a tree similar to the GUIDE algorithm
"""
import math
import sys
from enum import Enum
from parse import Settings, RegressionType, SplitPointMethod
from typing import List
from pprint import pprint
import logging
import heapq
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, chi2
from collections import defaultdict
from itertools import combinations, chain
from dill.source import getsource
import pandas as pd
from statsmodels.sandbox.stats.multicomp import multipletests

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('My Logger')

class TerminalData:
    def __init__(self, value):
        assert isinstance(value, float)
        self.value = value

class InternalData:
    def __init__(self, split_var, split_point, predicate, na_goes_left):
        self.split_point = split_point
        self.split_var = split_var
        self.predicate = predicate
        self.na_goes_left = na_goes_left


class Node:
    """ Regression tree """

    def __init__(self, type_specific_data, depth: int, parent, indices, node_num=1):
        
        assert isinstance(type_specific_data, TerminalData) or isinstance(type_specific_data, InternalData) or type_specific_data is None
        assert isinstance(depth, int)
        assert isinstance(parent, Node) or parent is None 
        assert isinstance(indices, np.ndarray)
        self.type_specific_data = type_specific_data
        self.left = None
        self.right = None
        self.depth = depth
        self.idx = indices
        self.node_num = node_num 

    def __str__(self):
        name = "Internal Node" if isinstance(self.type_specific_data, InternalData) else "Terminal Node"
        depth = "   "*self.depth
        desc = f"value = {self.type_specific_data.value} cnt = {self.idx.shape[0]}" if isinstance(self.type_specific_data, TerminalData) else \
                  f"pred = {self.type_specific_data.split_var} {self.type_specific_data.split_point}"
        return f"{depth} {name} {desc}"
        

class Model:
    """
    At each node, a constant (namely, the sample Y -mean) is ﬁtted and the residuals computed.
    To solve the ﬁrst problem,
    we use instead the Pearson chi-square test to detect associations between the
    signed residuals and groups of predictor values. If X is a c-category predictor,
    the test is applied to the 2 × c table formed by the two groups of residuals as
    rows and the categories of X as columns. If X is a numerical-valued variable, its
    values can be grouped to form the columns of the table. We divide the range of
    X into four groups at the sample quartiles to yield a 2 × 4 table. There are other
    ways to deﬁne the groups for ordered variables, but there is probably none that
    is optimal for all situations. Our experience indicates that this choice provides
    suﬃcient detection power while keeping the chance of empty cells low.

    @TODO: Skip the interaction tests for the first draft
    Although the chi-square test is sensitive to curvature along the direct, it does
    not work as well with with simple interaction models such as
    I(X1*X2 > 0) - I(X1*X2) <= 0) + noise


    So far we have concentrated on the problem of variable selection. To complete
    the tree construction algorithm, we need to select the split points as well as
    determine the size of the tree. For the latter, we adopt the CART method of
    cost-complexity pruning with an independent test set or, in its absence, by cross-
    validation.
    """
    def __init__(self, settings: Settings):
        self.df = settings.df
        self.tgt = settings.dependent_var
        self.weight_var = settings.weight_var
        self.split_vars = settings.split_vars
        self.col_data = settings.col_data
        self.split_point_method = SplitPointMethod.Greedy
        self.model_type = RegressionType.LINEAR_PIECEWISE_CONSTANT
        self.MIN_SAMPLES_LEAF = settings.MIN_SAMPLES_LEAF
        self.MAX_DEPTH = settings.MAX_DEPTH
        self.node_list = []
        self.idx_active = settings.idx_active
        self.top_node = Node(
            type_specific_data=InternalData(None, None, None, True),
            depth=0,
            parent=None,
            indices=self.idx_active)
        self.next_node_num = 1
        self.one_df_chi2_at_root = {}

    def _get_next_node_num(self):
        ret_val = self.next_node_num
        self.next_node_num = self.next_node_num + 1
        return ret_val
   

    def _calc_chi2_stat(self, node, y_mean, col) -> np.float64:
        """ Split numeric into 4 quartiles, split categoricals into c bins
        Calculate chi2_contingency and return p-value """
        # @NOTE: can we use pvalues as is or do I need to use modified wilson-hilferty to 
        # get the chi-squared degree 1 and rank the variables like that? The 2002 regression
        # paper says to sort based on p-values but the tutorial video part 1 says to rank
        # based on the chi-squared degree of freedom 1. Note the video is 20 years newer
        # than the paper.
        y_mean = self.df.loc[node.idx, self.tgt].mean()
        residuals = self.df.loc[node.idx, self.tgt] - y_mean
        # logger.log(level = logging.DEBUG, msg = f"idx_active size in chi2_stat = {len(node.idx)}")
        pvalue = sys.float_info.max 
        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # Convert the column to a NumPy array
                vals = self.df.loc[node.idx, col].values
                indexes = self.df.loc[node.idx, col].index.values
                
                # Bin the quartiles
                if (len(residuals) >= 60):
                    edges = np.percentile(
                        vals, [25, 50, 75, 100], method='linear')
                else:
                    edges = np.percentile(
                        vals, [100/3, 200/3, 300/3], method='linear')
                # Bin the data using np.digitize
                bins = np.digitize(
                    vals, edges, right=True)
                # Create a defaultdict to store grouped indexes
                grouped_indexes = defaultdict(list)

                # Iterate through the bins and indexes arrays
                for bin_value, index in zip(bins, indexes):
                    grouped_indexes[bin_value].append(index)

                grouped_index_keys = list(grouped_indexes.keys())
                num_groups = len(grouped_indexes.keys())
                chi_squared = np.zeros(shape=(2, num_groups))
                for _bin in range(0, num_groups):
                    chi_squared[0, _bin] = (
                        residuals[grouped_indexes[grouped_index_keys[_bin]]] > 0).sum()
                    chi_squared[1, _bin] = (
                        residuals[grouped_indexes[grouped_index_keys[_bin]]] <= 0).sum()

                # @TODO add column and row sum checks

                contingency_result = chi2_contingency(chi_squared, False)
                statistic = contingency_result.statistic
                dof = contingency_result.dof 
                one_dof_stat = wilson_hilferty(statistic, dof)
                # save the 1-df chi2 values at root node
                if node.node_num == 1:
                    self.one_df_chi2_at_root[col] = one_dof_stat
                pvalue = pvalue_for_one_dof(one_dof_stat) 
                return pvalue

            case 'c':
                # Specify the number of columns in the contingency table
                unique_vals = self.df.loc[self.idx_active, col].unique()  # includes NA
                num_cat = len(unique_vals)
                # Convert the column to a NumPy array
                vals = self.df.loc[self.idx_active, col].values
                indexes_by_value = self.df.loc[self.idx_active].groupby(col, dropna=False).apply(lambda group: group.index.values)

                chi_squared = np.zeros(shape=(2, num_cat))
                for _bin in range(0, num_cat):
                    chi_squared[0, _bin] = (
                        residuals[indexes_by_value[unique_vals[_bin]]] >= 0).sum()
                    chi_squared[1, _bin] = (
                        residuals[indexes_by_value[unique_vals[_bin]]] < 0).sum()
                pvalue = chi2_contingency(chi_squared).pvalue
            case _:
                raise f"split_var role not handled in {self.__name__}"
        return pvalue

    def _get_split_point_greedy(self, node, col):
        """ Get the optimal split value for a given split variable 
        Returns split_point, boolean for whether NA goes left
        split_point is numeric for a numeric column followed by a boolean
        split_point is a tuple of categories for categorical column followed by None

        G method is greedy exhaustive
        M method is median
        """

        _df = self.df.loc[node.idx, [col, self.tgt]]

        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # numeric
                x_uniq = _df[col].drop_duplicates().sort_values()
                cutpoints = x_uniq[:-1] + np.diff(x_uniq)/2
                greatest_tot_sse = -999
                best_cut = None
                node_sse = ((_df[self.tgt] - _df[self.tgt].mean())**2).sum()
                for cut in cutpoints:
                    right_idx = _df[_df[col] > cut].index
                    left_idx = _df.drop(right_idx, axis=0).index
                    left_mean = _df.loc[left_idx][self.tgt].mean()
                    right_mean = _df.loc[right_idx][self.tgt].mean()
                    nAL = len(left_idx)
                    nAR = len(right_idx)
                    tot_items = nAL + nAR 
                    cut_sse = (nAL * nAR / tot_items) * (left_mean - right_mean)**2
                    if cut_sse > greatest_tot_sse:
                        greatest_tot_sse = cut_sse
                        best_cut = cut
                
                return best_cut, False

            case 'c':
                # categorical
                """
                If X is a categorical predictor, we need to ﬁnd a split of the form X ∈ A,
                where A is a subset of the values taken by X. We accomplish this by viewing
                it as a classiﬁcation problem. Label each observation in the node as class 1
                if it is associated with a positive residual and as class 2 otherwise. Given a
                split determined by A, let L and R denote the data subsets in the left and
                right subnodes, respectively. We choose the set A for which the sum of the
                (binomial) variances in L and R is minimized. This solution is quickly found
                with an algorithm in Breiman et al. (1984, p.101).
                """
                x_uniq = _df[col].drop_duplicates().sort_values().values
                max_r = round(x_uniq.shape[0] / 1.5)
                max_r = min([max_r, 5]) 
                logger.log(level = logging.DEBUG, msg = f"finding best combination of categoricals with max_r = {max_r}")
                # avoid combinatorial explosion
                results = {'set' : [],'sum_binom_variance' : []}
                for subset in chain(*(combinations(x_uniq, r) for r in range(1, max_r + 1))):
                   positive_resid_idx = _df[_df[self.tgt] - node.y_mean > 0].index.values
                   negative_resid_idx = _df[_df[self.tgt] - node.y_mean < 0].index.values
                   left_idx = _df[_df[col].isin(subset)].index.values
                   right_idx = _df[~_df[col].isin(subset)].index.values
                   
                   class_1_left = np.intersect1d(left_idx, positive_resid_idx)
                   class_2_left = np.intersect1d(left_idx, negative_resid_idx)
                   
                   class_1_right = np.intersect1d(right_idx, positive_resid_idx)
                   class_2_right = np.intersect1d(right_idx, negative_resid_idx)
                   
                   probs_left = np.asarray([class_1_left.shape[0], class_2_left.shape[0]])   
                   probs_right = np.asarray([class_1_right.shape[0], class_2_right.shape[0]])
                   residual_impurity_left = probs_left[0]*probs_left[1]
                   residual_impurity_right = probs_right[0]*probs_right[1]

                   results['set'].append(subset)
                   results['sum_binom_variance'].append(residual_impurity_left + residual_impurity_left)
                    
                idx_min = np.argmin(results['sum_binom_variance'])
                return results['set'][idx_min], None

    def _get_split_point_median(self, node, col):
        """ Get the optimal split value for a given split variable 
        G method is greedy exhaustive
        M method is median
        """

        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # numeric
                return self.df.loc[node.idx, col].median()
            case 'c':
                # categorical
                # Based on 2002 paper it appears that categoricals are split the same
                # whether we are in Median or Greedy split point mode.
                return self._get_split_point_greedy(node, col)

    def _get_best_variable(self, node) -> str:
        """ Find best unbiased splitter among self.split_vars. 
            1. Curvature tests
            2. Interaction test (@todo)
        """
        if node.node_num == 2:
            pass
            # pdb.set_trace()
        
        if self.weight_var == list():
            node.y_mean = self.df.loc[node.idx, self.tgt].mean()
        else:
            node.y_mean = (self.df.loc[node.idx, self.tgt] *
                        self.df.loc[node.idx, self.weight_var]).sum() / self.df.loc[node.idx, self.weight_var].sum()
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean
        stat_pval = {
            col: self._calc_chi2_stat(
                node=node, 
                y_mean=node.y_mean,
                col=col) for col in self.split_vars}
        
        # Bonferonni correction (uses statsmodels)
        p_adjusted = multipletests([*stat_pval.values()], method='bonferroni')[1]
        stat_pval = { k : p_adjusted[idx] for idx, k in enumerate(stat_pval.keys())}

        top_3_keys = {key: value for key, value in stat_pval.items(
        ) if value in heapq.nsmallest(3, stat_pval.values())}
        top_3_keys = sorted(top_3_keys.items(), key=lambda x: x[1])
        if node == self.top_node:
            print()
            print("Top-ranked variables and 1-df chi-squared values at root node")
            for idx, (col, stat) in enumerate(self.one_df_chi2_at_root.items()):
                print(" "*5 + f"{idx+1} {stat:4.4E} {col}")
            print()
        return top_3_keys[0][0]

    def fit(self):
        """ Build model from training data """
        node_list = [None]*200 # all nodes of tree
        stack = [None]*200     # nodes that need processed
        stack.clear()
        node_list.clear()
        self.top_node.node_num = self._get_next_node_num()
        stack.append(self.top_node)
        
        assert self.model_type == RegressionType.LINEAR_PIECEWISE_CONSTANT
      
        
        # process nodes, adding new nodes as they are created
        
        while len(stack) > 0:
            curr = stack.pop(0)     
            # get split variable and split point
            na_left = None
            split_var = self._get_best_variable(node=curr)
            if self.split_point_method == SplitPointMethod.Greedy:
                split_point, na_left = self._get_split_point_greedy(node=curr, col=split_var)
            elif self.split_point_method == SplitPointMethod.Median:
                split_point, na_left = self._get_split_point_median(node=curr, col=split_var)
            elif self.split_point_method == SplitPointMethod.Systematic:
                raise "not implemented"

            if split_point == None:
                curr.type_specific_data = TerminalData(value = curr.y_mean)
                node_list.append(curr) 
                continue

            assert isinstance(curr.idx, np.ndarray)
            _df = self.df.loc[curr.idx]
            predicate = None
           
            # create predicate (lambda) for splitting dataframe
            # can be printed with:
            #   from dill.source import getsource
            if isinstance(split_point, tuple):
                predicate = lambda x,split_point=split_point : x in split_point
            else:
                if na_left == True:
                    predicate = lambda x,split_point=split_point : x < split_point or np.isnan(x)
                else:
                    predicate = lambda x,split_point=split_point : x < split_point
          
            # Split dataframe
            # @NOTE can index a dataframe by a boolean but need to call .loc to index it with an index
            left = _df[_df[split_var].map(predicate)].index.values
            right = _df[~_df[split_var].map(predicate)].index.values
            assert left.shape[0] + right.shape[0] == curr.idx.shape[0]

            if left.shape[0] < self.MIN_SAMPLES_LEAF or right.shape[0] < self.MIN_SAMPLES_LEAF \
                    or curr.depth == self.MAX_DEPTH:
                        # Based on early stopping, make curr node a leaf 
                        curr.type_specific_data = TerminalData(value = curr.y_mean)
                        node_list.append(curr) 
                        continue
           
            # Terminate left or right node if they are homogonous
            assert predicate is not None
            curr.type_specific_data=InternalData(split_var=split_var, split_point=split_point, \
                    predicate=predicate, na_goes_left=na_left)

            # Split node
            left_node = Node(type_specific_data=None, depth = curr.depth + 1, parent=curr, indices=left, node_num=self._get_next_node_num())
            right_node = Node(type_specific_data=None, depth = curr.depth + 1, parent=curr, indices=right, node_num=self._get_next_node_num())
            curr.left = left_node
            curr.right = right_node
            stack.append(left_node)
            stack.append(right_node)
            node_list.append(curr)

    def _print_tree(self, node, depth):
        """ recursively print out the tree like the reference output """
        # @TODO: Add support for categoricals
        spacer = "  "
        # base case terminal node
        if node.left == None and node.right == None:
            print((depth-1)*spacer + f"Node {node.node_num} : target-mean = {node.type_specific_data.value:9f} ({len(node.idx)})")
            return
        # print left branch
        print((depth-1)*spacer + f"Node {node.node_num}: {node.type_specific_data.split_var} <= {node.type_specific_data.split_point:9f} ({len(node.idx)})")
        self._print_tree(node.left, depth+1)
        # print right branch
        print((depth-1)*spacer + f"Node {node.node_num}: {node.type_specific_data.split_var} > {node.type_specific_data.split_point:9f} ({len(node.idx)})")
        self._print_tree(node.right, depth+1)

    def print(self):
        curr_node = self.top_node
        curr_depth = 1
        self._print_tree(curr_node, curr_depth)



    def _prune(node):
        """
        Unless stated otherwise, the trees presented here are pruned using the cost-
        complexity pruning method of CART with N -fold cross-validation, where N is
        the size of the training sample. That is, an overly large tree is constructed and
        then sequentially pruned back until only the root node is left. This yields a
        sequence of nested subtrees. The prediction mean square error (PMSE) of each
        subtree is estimated by N -fold cross-validation. The subtree with the smallest
        PMSE (p0 , say) is called the 0-SE tree. Letting s0 be the estimated standard
        error of p0 , the 1-SE tree is the smallest subtree whose estimated PMSE is less
        than p0 + s0 . The reader is referred to Breiman et al. (1984, Sec. 3.4) for further
        details on pruning and estimation of standard error.
        """
        pass

    def predict(self, test):
        predictions = [] 
        idxs = []
        """ Generate model predictions """
        for idx, row in test.iterrows():
            curr = self.top_node    
            # Get to leaf node
            while isinstance(curr.type_specific_data, InternalData):
                feat = curr.type_specific_data.split_var
                predicate = curr.type_specific_data.predicate
                goes_left = predicate(row[feat]) 
                if type(goes_left) != np.bool_ and type(goes_left) != bool:
                    pdb.set_trace()
                if goes_left == True:
                    curr = curr.left
                else:
                    curr = curr.right

            # add leaf value to predictions
            assert isinstance(curr.type_specific_data, TerminalData)
            predictions.append(curr.type_specific_data.value)
            idxs.append(idx)
        
        assert test.shape[0] == len(predictions) and test.shape[0] == len(idxs)

        return pd.DataFrame(index=idxs, columns=['pred'],data=predictions)
    
        """ Predict the model results for a test dataframe. 
        self is the top node of the tree.  
        predictions = []
        row_idx = -1 
        while len(predictions) < df.shape[0]:
            row_idx += 1
            curr_node = self
            curr_row = df.iloc[row_idx]
            while curr_node.type == 'node':
                curr_feat = curr_node.feature
                curr_cutpoint = curr_node.cutpoint
                if curr_row[curr_feat] >= curr_cutpoint:
                    curr_node = curr_node.left
                else:
                    curr_node = curr_node.right
            # once made it here, we should be at a leaf
            assert curr_node.type == 'leaf'
            predictions.append(curr_node.value)
        return np.asarray(predictions)
        """
        return predictions


#####################################################
# Helper functions
#####################################################
def _sse(vals: np.ndarray):
    """ calc sum of squares """
    return np.sum(vals**2)

def wilson_hilferty(stat, dof) -> np.float64:
    """ approximately convert chi-squared with dof degrees of freedom to 1 degree of freedom """
    w1 = (math.sqrt(2*stat) - math.sqrt(2*dof - 1) + 1)**2 / 2
    temp = 7/9 + math.sqrt(dof)*( (stat/dof)**(1/3) - 1 + 2 / (9 * dof) )
    w2 = max(0, temp ** 3) 

    w = None
    if stat < dof + 10 * math.sqrt(2*dof):
        w = w2
    elif stat >= dof + 10 * math.sqrt(2*dof) and w2 < stat:
        w = (w1 + w2) / 2
    else:
        w = w1
    return w

def pvalue_for_one_dof(stat):
    """ pvalue for a 1 dof chi squared statistic """
    return 1 - chi2.cdf(stat, 1)
