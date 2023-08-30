"""
1. Unbiased variable selection with curvature test and interaction test
1.

"""

from enum import Enum
from parse import Settings
from typing import List
from pprint import pprint
import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from collections import defaultdict


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

    def residuals(self):
        pass


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

    def _calc_chi2_stat(self, y_mean, col) -> np.float64:
        """ Split numeric into 4 quartiles, split categoricals into c bins
        Calculate chi2_contingency and return p-value """
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
                unique_vals = self.df[col].unique() # includes NA
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

    def _get_best_split(self):
        sample_y_mean = (self.df.loc[self.top_node.idx, self.tgt] *
                         self.df[self.weight_var]).sum() / self.df[self.weight_var].sum()
        residuals = self.df.loc[self.top_node.idx, self.tgt] - sample_y_mean
        print(f"sample y_mean {sample_y_mean}")

        stat_pval = {
            col: self._calc_chi2_stat(
                y_mean=sample_y_mean,
                col=col) for col in self.split_vars}
        # numerical val          |  0   0.25% | 0.25 to 0.50 | 0.50 to 0.75 | 0.75 - 1.0 |
        #                   pos
        #                   neg

        # categorical val
        #                        |   cat1     |   cat2       |    cat3      |   NA       | ... etc
        #                   pos
        #                   neg
        sorted_items = sorted(stat_pval.items(), key=lambda item: item[1])
        pprint(sorted_items[:10])

    def fit(self):
        """ Build model from training data """
        best_split_var = self._get_best_split()

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
