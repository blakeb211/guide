"""
1. Parse description and data file
1. Pre process (switching columns, dropping rows)
1. Unbiased variable selection with curvature test and interaction test 
1. 
"""


class NodeData:
    """ Info for each node to hold """


class Node:
    """ Some version of a Regression tree """

    def __init__(self, params):
        """ Construct with hyperparameters  """
        pass

    def fit(self, train):
        """ Build model from training data """

        """
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

    def predict(self, test):
        """ Generate model predictions """
        pass



def prune(node):
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

