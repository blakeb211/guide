"""
Build a tree similar to the GUIDE algorithm
"""

from enum import Enum
from parse import Settings, RegressionType, SplitPointMethod
from typing import List
from pprint import pprint
import heapq
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from collections import defaultdict
from itertools import permutations, chain

class NodeType(Enum):
    Internal = 1,
    Terminal = 2


class TerminalData:
    def __init__(self, value, samples: List[int]):
        self.value = value
        self.samples = []


class InternalData:
    def __init__(self, split_var, cutpoint):
        self.cutpoint = cutpoint
        self.split_var = split_var
        self.NA_goes_right = True


class Node:
    """ Regression tree """

    def __init__(self, node_type: NodeType, depth: int, indices):
        self.type = node_type
        self.left = None
        self.right = None
        self.depth = depth
        self.idx = indices


class Model:
    def __init__(self, settings: Settings):
        self.df = settings.df
        self.tgt = settings.dependent_var
        self.weight_var = settings.weight_var
        self.split_vars = settings.split_vars
        self.col_data = settings.col_data
        self.top_node = Node(
            node_type=NodeType.Internal,
            depth=0,
            indices=self.df.index.values)
        self.split_point_method = SplitPointMethod.Greedy
        self.model_type = RegressionType.LINEAR_PIECEWISE_CONSTANT

    def _calc_chi2_stat(self, y_mean, col) -> np.float64:
        """ Split numeric into 4 quartiles, split categoricals into c bins
        Calculate chi2_contingency and return p-value """
        # @NOTE: can we use pvalues as is or do I need to use modified wilson-hilferty to 
        # get the chi-squared degree 1 and rank the variables like that? The 2002 regression
        # paper says to sort based on p-values but the tutorial video part 1 says to rank
        # based on the chi-squared degree of freedom 1. Note the video is 20 years older
        # than the paper.
        residuals = self.df[self.tgt] - y_mean
        pvalue = 999.99
        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # Specify the number of quartiles
                num_quartiles = 4
                # Convert the column to a NumPy array
                column_array = self.df[col].values
                indexes = self.df[col].index.values
                # Bin the quartiles
                quartile_edges = np.percentile(
                    column_array, [25, 50, 75, 100], method='linear')
                # Bin the data using np.digitize
                quartile_bins = np.digitize(
                    column_array, quartile_edges, right=True)
                # Create a defaultdict to store grouped indexes
                grouped_indexes = defaultdict(list)

                # Iterate through the bins and indexes arrays
                for bin_value, index in zip(quartile_bins, indexes):
                    grouped_indexes[bin_value].append(index)

                grouped_index_keys = list(grouped_indexes.keys())
                num_groups = len(grouped_indexes.keys())
                chi_squared = np.zeros(shape=(2, num_groups))
                for _bin in range(0, num_groups):
                    chi_squared[0, _bin] = (
                        residuals[grouped_indexes[grouped_index_keys[_bin]]] >= 0).sum()
                    chi_squared[1, _bin] = (
                        residuals[grouped_indexes[grouped_index_keys[_bin]]] < 0).sum()
                pvalue = chi2_contingency(chi_squared).pvalue
            case 'c':
                # Specify the number of columns in the contingency table
                unique_vals = self.df[col].unique()  # includes NA
                num_cat = len(unique_vals)
                # Convert the column to a NumPy array
                column_array = self.df[col].values
                indexes_by_value = self.df.groupby(col, dropna=False).apply(lambda group: group.index.values)

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
        G method is greedy exhaustive
        M method is median
        """
        _df = self.df.loc[node.idx, [col, self.tgt]]
        if _df[col].isnull().sum() > 0:
            pdb.set_trace()

        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # numeric
                x_uniq = _df[col].drop_duplicates().sort_values()
                cutpoints = x_uniq[:-1] + np.diff(x_uniq)/2
                smallest_tot_sse = None # total weighted sse of the node - left - right
                cut_with_smallest_sse = None
                for cut in cutpoints:
                    right_idx = _df[_df[col] < cut].index.values
                    left_idx = _df[_df[col] >= cut].index.values
                    tot_items = len(right_idx) + len(left_idx)
                    weights = tot_items / tot_items, len(right_idx)/tot_items, len(left_idx)/tot_items
                    left_resid = _df.loc[left_idx, self.tgt] - _df.loc[left_idx, self.tgt].mean()
                    right_resid = _df.loc[right_idx, self.tgt] - _df.loc[right_idx, self.tgt].mean()
                    node_resid = _df[self.tgt].values - node.y_mean
                    tot_sse = weights[0]*_sse(node_resid) - weights[1] * \
                        _sse(right_resid) - weights[2]*_sse(left_resid)
                    if smallest_tot_sse == None or smallest_tot_sse > tot_sse:
                        smallest_tot_sse = tot_sse
                        cut_with_smallest_sse = cut
                return cut_with_smallest_sse
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
                # avoid combinatorial explosion
                max_r = max([max_r, 8]) 
                results = {'set','sum_binom_variance'}
                for subset in chain(*(permutations(x_uniq, r) for r in range(1, max_r + 1))):
                   left_idx = _df[_df[col].isin(subset)].index.values
                   right_idx = _df[~_df[col].isin(subset)].index.values

    def _get_split_point_median(self, node, col):
        """ Get the optimal split value for a given split variable 
        G method is greedy exhaustive
        M method is median
        """
        if self.df[col].isnull().sum() > 0:
            pdb.set_trace()

        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
            case 'S':
                # numeric
                return self.df.loc[node.idx, col].median()
            case 'c':
                # categorical
                # Based on 2002 paper it appears that categoricals are split the same
                # whether we are in Median or Greedy split point mode.
                return self._get_split_point_greedy(node, col)

    def _get_best_split(self, node) -> str:
        """ Find best unbiased splitter among self.split_vars. """
        # @TODO: Add interaction tests
        node.y_mean = (self.df.loc[node.idx, self.tgt] *
                       self.df.loc[node.idx, self.weight_var]).sum() / self.df.loc[node.idx, self.weight_var].sum()
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean
        stat_pval = {
            col: self._calc_chi2_stat(
                y_mean=node.y_mean,
                col=col) for col in self.split_vars}
        # numerical val          |  0   0.25% | 0.25 to 0.50 | 0.50 to 0.75 | 0.75 - 1.0 |
        #                   pos
        #                   neg

        # categorical val
        #                        |   cat1     |   cat2       |    cat3      |   NA       | ... etc
        #                   pos
        #                   neg
        top_3_keys = {key: value for key, value in stat_pval.items(
        ) if value in heapq.nsmallest(3, stat_pval.values())}
        top_3_keys = sorted(top_3_keys, key=lambda x: x[1])
        # hopefully heapq is faster!
        # sorted_items = sorted(stat_pval.items(), key=lambda item: item[1])
        col = top_3_keys[0]
        return top_3_keys[0]

    def fit(self):
        """ Build model from training data """
        if self.model_type == RegressionType.LINEAR_PIECEWISE_CONSTANT:
            best_split_var = self._get_best_split(node=self.top_node)
            # @TODO: REMOVE
            best_split_var = 'CUTENURE'
    
        if self.split_point_method == SplitPointMethod.Greedy:
            split_point = self._get_split_point_greedy(node=self.top_node, col=best_split_var)
        elif self.split_point_method == SplitPointMethod.Median:
            split_point = self._get_split_point_median(node=self.top_node, col=best_split_var)
        elif self.split_point_method == SplitPointMethod.Systematic:
            raise "not implemented"

        print(f"best split = {best_split_var, split_point}")

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

    def predict(self, test):
        """ Generate model predictions """
        pass


def _sse(vals: np.ndarray):
    """ calc sum of squares """
    return np.sum(vals**2)
