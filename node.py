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
    """ Node class """

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
        self.tree_text = [] 

    def _get_next_node_num(self):
        ret_val = self.next_node_num
        self.next_node_num = self.next_node_num + 1
        return ret_val
   


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
                node_sse = ((_df[self.tgt] - node.y_mean)**2).sum()
                sse_tab = pd.DataFrame({'cut':[],'sse':[]})

                for idx, cut in enumerate(cutpoints):
                    right_idx = _df[_df[col] > cut].index
                    left_idx = _df.drop(right_idx, axis=0).index
                    left_mean = _df.loc[left_idx][self.tgt].mean()
                    right_mean = _df.loc[right_idx][self.tgt].mean()


                    
                    left_sse = ( (_df.loc[left_idx, self.tgt]- _df.loc[left_idx, self.tgt].mean())**2 ).sum()
                    right_sse = ( (_df.loc[right_idx, self.tgt]- _df.loc[right_idx, self.tgt].mean())**2 ).sum()
                    sse_sum2 = len(left_idx)*left_sse + len(right_idx)*right_sse

                    

                    nAL = len(left_idx)
                    nAR = len(right_idx)
                    tot_items = nAL + nAR 
                    cut_sse = (nAL * nAR / tot_items) * (left_mean - right_mean)**2
                    sse_tab = pd.concat([sse_tab,pd.DataFrame({'cut':cut,'sse':cut_sse, 'sse_sum2':sse_sum2},index=[idx])])


                    if cut_sse > greatest_tot_sse and len(right_idx) >= self.MIN_SAMPLES_LEAF and len(left_idx) >= self.MIN_SAMPLES_LEAF:
                        greatest_tot_sse = cut_sse
                        best_cut = cut
            
                if False and node.node_num == 14:
                    pdb.set_trace()
                

                # returns None if no cutpoint found
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
                max_r = int(round(x_uniq.shape[0] / 2 - 0.1))
                max_r = max(max_r, 1)
                # logger.log(level = logging.DEBUG, msg = f"finding best {max_r} of {len(x_uniq)} category combos")
                
                results = {'set' : [],'gain' : []}
                subsets = [subset for subset in chain(*(combinations(x_uniq, r) for r in range(1, max_r + 1)))]
                
                for subset in subsets:
                   left_idx = _df[_df[col].isin(subset)].index.values
                   right_idx = _df.drop(left_idx).index.values

                   # gini impurity of left and right nodes based on residual sign
                   mean_left = (_df.loc[left_idx, self.tgt] <= node.y_mean).mean() 
                   mean_right = (_df.loc[right_idx, self.tgt] <= node.y_mean).mean() 
                   mean_node  = (_df[self.tgt] <= node.y_mean).mean()

                   Nall = node.idx.shape[0]
                   p = 1, len(left_idx)/Nall, len(right_idx)/Nall
                    
                   gini_node = 2 * mean_node * (1 - mean_node)
                   gini_left = 2 * mean_left * (1 - mean_left)
                   gini_right = 2 * mean_right * (1 - mean_right)
                   gain = p[0]*gini_node - p[1]*gini_left - p[2]*gini_right
                   gain = round(gain,10)
    
                   results['set'].append(subset)
                   results['gain'].append(gain)


                idx_max = np.argmax(results['gain'])
                return results['set'][idx_max], None

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


    def _calc_chi2_stat(self, node, col) -> np.float64:
        """ Split numeric into 4 quartiles, split categoricals into c bins
        Calculate chi2_contingency and return p-value """
        # @NOTE: can we use pvalues as is or do I need to use modified wilson-hilferty to 
        # get the chi-squared degree 1 and rank the variables like that? The 2002 regression
        # paper says to sort based on p-values but the tutorial video part 1 says to rank
        # based on the chi-squared degree of freedom 1. Note the video is 20 years newer
        # than the paper.

        y_mean = self.df.loc[node.idx, self.tgt].mean()
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean 
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

                # Column and row sum checks
                # @TODO remove rows or columns is they are empty
                column_sums = np.sum(chi_squared, axis=0)
                row_sums = np.sum(chi_squared, axis=1)
                assert np.sum(column_sums == 0) == 0
                assert np.sum(row_sums == 0) == 0

                contingency_result = chi2_contingency(chi_squared, False)
                statistic = contingency_result.statistic
                dof = contingency_result.dof 
           
                # statistic == 0 breaks wilson_hilferty
                if abs(statistic - 0) > 1E-7:
                    one_dof_stat = wilson_hilferty(statistic, dof)
                    pvalue = pvalue_for_one_dof(one_dof_stat) 
                else:
                    one_dof_stat = 0.0
                    pvalue = 1.0 

                # save the 1-df chi2 values at root node
                if node.node_num == 1:
                    self.one_df_chi2_at_root[col] = one_dof_stat
                    # statistic == 0 breaks wilson_hilferty

                return pvalue

            case 'c':
                logger.log(logging.DEBUG, msg = f"running _calc_chi2_stat, categorical section, col = {col}, node size = {len(node.idx)}")
                # Specify the number of columns in the contingency table
                unique_vals = self.df.loc[node.idx, col].unique()  # includes NA
                num_cat = len(unique_vals)
                # Convert the column to a NumPy array
                
                indexes_by_value = self.df.loc[node.idx].groupby(col, dropna=False).apply(lambda group: group.index.values)
                
                chi_squared = np.zeros(shape=(2, num_cat))
                for _bin in range(0, num_cat):
                    chi_squared[0, _bin] = (
                        residuals[indexes_by_value[unique_vals[_bin]]] >= 0).sum()
                    chi_squared[1, _bin] = (
                        residuals[indexes_by_value[unique_vals[_bin]]] < 0).sum()
                
                # Column and row sum checks
                # @TODO remove rows or columns is they are empty
                column_sums = np.sum(chi_squared, axis=0)
                row_sums = np.sum(chi_squared, axis=1)
                assert np.sum(column_sums == 0) == 0
                assert np.sum(row_sums == 0) == 0

                contingency_result = chi2_contingency(chi_squared, False)
                statistic = contingency_result.statistic
                dof = contingency_result.dof 
           
                # statistic == 0 breaks wilson_hilferty
                if abs(statistic - 0) > 1E-7:
                    one_dof_stat = wilson_hilferty(statistic, dof)
                    pvalue = pvalue_for_one_dof(one_dof_stat) 
                else:
                    one_dof_stat = 0.0
                    pvalue = 1.0 

                # save the 1-df chi2 values at root node
                if node.node_num == 1:
                    self.one_df_chi2_at_root[col] = one_dof_stat
                    # statistic == 0 breaks wilson_hilferty

            case _:
                raise f"split_var role not handled in {self.__name__}"
        return pvalue


    def _get_best_variable(self, node) -> str:
        """ Find best unbiased splitter among self.split_vars. 
            1. Curvature tests
            2. Interaction test (@todo)
        """
        
        if self.weight_var == list():
            node.y_mean = self.df.loc[node.idx, self.tgt].mean()
        else:
            node.y_mean = (self.df.loc[node.idx, self.tgt] *
                        self.df.loc[node.idx, self.weight_var]).sum() / self.df.loc[node.idx, self.weight_var].sum()
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean
        stat_pval = {
            col: self._calc_chi2_stat(
                node=node, 
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
            for idx, (col, stat) in enumerate(sorted(self.one_df_chi2_at_root.items(),key=lambda x_y:x_y[1], reverse=True)):
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

            if False and curr.node_num == 14:
                pdb.set_trace()
            
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
            logger.log(logging.DEBUG, msg = f"splitting node {curr.node_num} with split_var = {split_var} at split point = {split_point}")
            left = _df[_df[split_var].map(predicate)].index.values
            right = _df[~_df[split_var].map(predicate)].index.values
            assert left.shape[0] + right.shape[0] == curr.idx.shape[0]

            
            if left.shape[0] <= self.MIN_SAMPLES_LEAF or right.shape[0] <= self.MIN_SAMPLES_LEAF \
                    or curr.depth == self.MAX_DEPTH:
                        # Based on early stopping, make curr node a leaf 
                        curr.type_specific_data = TerminalData(value = curr.y_mean)
                        node_list.append(curr) 
                        continue
           
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

        # generate the tree text
        curr_node = self.top_node
        curr_depth = 1
        self._print_tree(curr_node, curr_depth)

    def _print_tree(self, node, depth):
        """ saves tree text to Model class.  recursively process the tree to match the reference output """
        # @TODO: Add support for categoricals
        spacer = "  "
        # base case terminal node
        if node.left == None and node.right == None:
            sn = (depth-1)*spacer + f"Node {node.node_num}: target-mean = {node.type_specific_data.value:9f}"
            self.tree_text.append(sn)
            return
        
        # symbols for categorical versus numeric
        # print left branch
        split_point_numeric = isinstance(node.type_specific_data.split_point, (int, float))
        left_sym = None
        right_sym = None
        if split_point_numeric:
            left_sym = '<='
            right_sym = '>'
        else:
            left_sym = '='
            right_sym = '/='

        # make split point printed output match ref output with quotes around categories separated by commas
        printable_split_point = ""
        if isinstance(node.type_specific_data.split_point, tuple):
            printable_split_point = ", ".join(["\""+str(i)+"\"" for idx,i in enumerate(node.type_specific_data.split_point)]) 
        else:
            printable_split_point = node.type_specific_data.split_point
        

        sl = (depth-1)*spacer + f"Node {node.node_num}: {node.type_specific_data.split_var} {left_sym} {printable_split_point}"
        self.tree_text.append(sl)
        self._print_tree(node.left, depth+1)
        # print right branch
        sr = (depth-1)*spacer + f"Node {node.node_num}: {node.type_specific_data.split_var} {right_sym} {printable_split_point}"
        self.tree_text.append(sr)
        self._print_tree(node.right, depth+1)
        

    def print(self):
        pass

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

    def predict_train_data(self, print_me=False) -> pd.DataFrame:
        """ Generate model predictions on train data equivalent to the data.node file """
        predictions = pd.DataFrame(columns=["train", "node", "observed", "predicted"])

        for idx, row in self.df.iterrows(): 
            curr = self.top_node    
            train = 'n'
            node = None 
            observed = None 
            predicted = None 

            if idx in self.top_node.idx:
                train = 'y'
            
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
                if isinstance(curr.type_specific_data, TerminalData):
                    node = curr.node_num
                    observed = row[self.tgt]
                    predicted = self.df.loc[curr.idx, self.tgt].mean()

            df2 = pd.DataFrame({'train' : train, 'node' : node, 'observed' : observed, 'predicted' : predicted}, index=[idx])
            predictions = pd.concat([predictions,df2])

        if print_me == True:
            print()
            print(predictions.to_string(index=False))
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
