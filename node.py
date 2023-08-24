class NodeData:
    """ Info for each node to hold """


class Node:
    """ Some version of a Regression tree """

    def __init__(self, params):
        """ Construct with hyperparameters  """
        pass

    def fit(self, train):
        """ Build model from training data """
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
