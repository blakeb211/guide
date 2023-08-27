"""
1. Unbiased variable selection with curvature test and interaction test
1.

Table: Variable role meanings
| Abbreviation | Meaning                                                   |
|--------------|-----------------------------------------------------------|
| d            | Dependent variable                                       |
| b            | Split and fit categorical variable using indicator variables |
| c            | Split-only categorical variable                         |
| i            | Fit-only categorical variable via indicators            |
| s            | Split-only numerical variable                           |
| n            | Split and fit numerical variable                        |
| f            | Fit-only numerical variable                             |
| m            | Missing-value flag variable                              |
| p            | Periodic variable                                       |
| w            | Weight                                                   |
"""

from enum import Enum
from parse import Settings
from typing import List
import matplotlib.pyplot as plt

class NodeType(Enum):
    Internal = 1,
    Terminal = 2

class TerminalData:
    def __init__(self, value, samples : List[int]):
        self.value = value
        self.samples = []

class InternalData:
    def __init__(self, split_var, cutpoint):
        self.cutpoint = cutpoint
        self.split_var = split_var
        self.NA_goes_right = True
       
class Node:
    """ Regression tree """ 
    def __init__(self, node_type : NodeType, depth : int, indices):
        self.type = node_type
        self.left = None
        self.right = None
        self.depth = depth
        self.idx = indices

    def residuals(self):
        pass

class Model:
    def __init__(self, settings : Settings):
        self.settings = settings
        self.top_node = Node(
                node_type=NodeType.Internal, 
                depth=0, 
                indices=self.settings.df.index.tolist())

    def _get_best_split(self):
        sample_y_mean = self.settings.df.loc[self.top_node.idx, self.settings.dependent_var].mean()
        residuals = self.settings.df.loc[self.top_node.idx, self.settings.dependent_var] - sample_y_mean
        plt.scatter(x=residuals.index.tolist(), y=residuals)
        plt.ylim(residuals.min(),residuals.max())
        print(f"residuals min = {residuals.min()}")
        plt.show()
        print(f"sample y_mean {sample_y_mean}")
        

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

