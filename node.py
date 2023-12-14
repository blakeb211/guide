"""
Build GUIDE-compatible tree models
"""
import math
import random
import hashlib
from parse import Settings, RegressionType, SplitPointMethod, parse_data
from typing import List
import logging
import pdb
import numpy as np
from scipy.stats import chi2_contingency, chi2
from collections import defaultdict
from itertools import combinations, product
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

    def __init__(self, node_data, depth: int, parent, indices, node_num=1):
        assert isinstance(node_data, TerminalData) or \
            isinstance(node_data, InternalData) or \
            node_data is None

        assert isinstance(depth, int)
        assert isinstance(parent, Node) or parent is None
        assert isinstance(indices, np.ndarray)
        self.node_data = node_data
        self.left = None
        self.right = None
        self.depth = depth
        self.idx = indices
        self.node_num = node_num

    def __str__(self):
        is_internal = isinstance(self.node_data, InternalData)
        name = "Internal Node" if is_internal else "Terminal Node"
        depth = "   " * self.depth
        desc = f"value = {self.node_data.value} cnt = {self.idx.shape[0]}" \
            if not is_internal else f"pred = {self.node_data.split_var}" + \
            " {self.node_data.split_point}"
        return f"{depth} {name} {desc}"


class Model:
    """
    GUIDE-like model class
    """

    def __init__(self, settings: Settings):
        parse_data(settings)
        self.df = settings.df
        self.tgt = settings.dependent_var
        self.weight_var = settings.weight_var
        self.split_vars = settings.split_vars
        self.col_data = settings.col_data
        self.roles = {var: self.col_data[self.col_data['var_name']
                                         == var].var_role.values[0] for var in self.split_vars}
        self.split_point_method = SplitPointMethod.Greedy
        self.model_type = RegressionType.PIECEWISE_CONSTANT
        self.MIN_SAMPLES_LEAF = settings.MIN_SAMPLES_LEAF
        self.MAX_DEPTH = settings.MAX_DEPTH
        self.node_list = []
        self.idx_active = settings.idx_active
        self.interactions_on = settings.interactions_on
        self.top_node_best_var = None
        self.top_node = Node(
            node_data=InternalData(None, None, None, True),
            depth=0,
            parent=None,
            indices=self.idx_active)
        self.next_node_num = 1
        self.one_df_chi2_at_root = {}
        self.tree_text = []
        self.settings = settings

    def _get_next_node_num(self):
        """ Get a node_number for a new node """
        ret_val = self.next_node_num
        self.next_node_num = self.next_node_num + 1
        return ret_val

    def _get_split_point_greedy(self, node, col):
        """ Get the optimal split value for a given split variable
        Returns split_point, boolean for whether NA goes left
        split_point is numeric for a numeric column followed by a boolean
        split_point is a tuple of categories for categorical column followed by None
        """
        _df = self.df.loc[node.idx, [col, self.tgt]]
        match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:

            case 'S':
                # numeric
                x_uniq = _df[col].drop_duplicates().sort_values()
                if x_uniq.shape[0] == 1:
                    # node already pure so should be Terminal node
                    return None, False
                cutpoints = x_uniq[:-1] + np.diff(x_uniq) / 2
                greatest_tot_sse = -999
                best_cut = None
                sse_tab = pd.DataFrame({'cut': [], 'sse': []})

                for idx, cut in enumerate(cutpoints):
                    right_idx = _df[_df[col] > cut].index
                    left_idx = _df.drop(right_idx, axis=0).index
                    left_mean = _df.loc[left_idx][self.tgt].mean()
                    right_mean = _df.loc[right_idx][self.tgt].mean()

                    nAL = len(left_idx)
                    nAR = len(right_idx)
                    tot_items = nAL + nAR
                    cut_sse = (nAL * nAR / tot_items) * \
                        (left_mean - right_mean)**2
                    sse_tab = pd.concat([sse_tab, pd.DataFrame(
                        {'cut': cut, 'sse': cut_sse}, index=[idx])])

                    if cut_sse > greatest_tot_sse \
                            and len(right_idx) >= self.MIN_SAMPLES_LEAF \
                            and len(left_idx) >= self.MIN_SAMPLES_LEAF:
                        greatest_tot_sse = cut_sse
                        best_cut = cut

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
                if x_uniq.shape[0] == 1:
                    # node already pure so should be Terminal node
                    return None, False

                results = {'set': [], 'gain': []}

                # 1984 Breiman book pg 101. Reduce 2^L splits to L-1 splits to
                # evaluate

                _df.loc[:, self.tgt] = _df[self.tgt] < node.y_mean

                def pi(j):
                    """ prior probability of the class in the node """
                    if j == 1:
                        return _df[self.tgt].mean()
                    if j == 0:
                        return 1.0 - pi(1)
                    raise "fail"

                def N_j_l(j, s):
                    """ num cases of class j in subset s"""
                    assert j == 0 or j == 1
                    return _df[(_df[self.tgt] == j) & (_df[col] == s)].shape[0]

                def p(j, subset_l) -> float:
                    """ When we sort categories by this value we get L-1 split points """
                    assert j == 0 or j == 1
                    divisor = (pi(1) * N_j_l(1, subset_l) +
                               pi(0) * N_j_l(0, subset_l))
                    return pi(j) * N_j_l(j, subset_l) / divisor

                sorted_ps = sorted([(s, p(1, s))
                                   for s in x_uniq], key=lambda x: x[1])
                sorted_ps = [s for s, _ in sorted_ps]
                end = 0
                subsets = [tuple(sorted_ps[0:end])
                           for end in range(1, len(x_uniq))]

                for subset in subsets:
                    left_idx = _df[_df[col].isin(subset)].index.values
                    right_idx = _df.drop(left_idx).index.values

                    # gini impurity of left and right nodes based on residual
                    # sign
                    mean_left = _df.loc[left_idx, self.tgt].mean()
                    mean_right = _df.loc[right_idx, self.tgt].mean()
                    mean_node = _df[self.tgt].mean()

                    Nall = node.idx.shape[0]
                    p = 1, len(left_idx) / Nall, len(right_idx) / Nall

                    gini_node = 2 * mean_node * (1 - mean_node)
                    gini_left = 2 * mean_left * (1 - mean_left)
                    gini_right = 2 * mean_right * (1 - mean_right)
                    gain = p[0] * gini_node - p[1] * \
                        gini_left - p[2] * gini_right
                    gain = round(gain, 10)

                    results['set'].append(subset)
                    results['gain'].append(gain)

                if len(results['gain']) == 0:
                    pdb.set_trace()

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

    def curvature_test(self, node) -> dict:
        """ Split numeric into 4 quartiles, split categoricals into c bins
        Calculate chi2_contingency return one_dof_stat dictionary """
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean
        ret_one_dof_stat = {}
        for col in self.split_vars:
            match self.col_data[self.col_data.var_name == col]['var_role'].iloc[0]:
                case 'S' | 'n':
                    # Convert the column to a NumPy array
                    vals = self.df.loc[node.idx, col].values
                    indexes = self.df.loc[node.idx, col].index.values

                    # Bin the quartiles
                    if len(residuals) >= 60:
                        edges = np.percentile(
                            vals, [25, 50, 75, 100], method='linear')
                    else:
                        edges = np.percentile(
                            vals, [100 / 3, 200 / 3, 300 / 3], method='linear')
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

                    # Remove columns is they are empty
                    chi_squared = remove_empty_cols(chi_squared)

                    contingency_result = chi2_contingency(chi_squared, False)
                    statistic = contingency_result.statistic
                    dof = contingency_result.dof

                    one_dof_stat = wilson_hilferty(statistic, dof)

                    ret_one_dof_stat[col] = one_dof_stat

                case 'c':
                    # Specify the number of columns in the contingency table
                    unique_vals = self.df.loc[node.idx,
                                              col].unique()  # includes NA
                    num_cat = len(unique_vals)
                    # Convert the column to a NumPy array

                    indexes_by_value = self.df.loc[node.idx].groupby(
                        col, dropna=False). apply(lambda group: group.index.values)

                    chi_squared = np.zeros(shape=(2, num_cat))
                    for _bin in range(0, num_cat):
                        chi_squared[0, _bin] = (
                            residuals[indexes_by_value[unique_vals[_bin]]] >= 0).sum()
                        chi_squared[1, _bin] = (
                            residuals[indexes_by_value[unique_vals[_bin]]] < 0).sum()
                    # remove cols if they are empty
                    chi_squared = remove_empty_cols(chi_squared)

                    contingency_result = chi2_contingency(chi_squared, False)
                    statistic = contingency_result.statistic
                    dof = contingency_result.dof

                    one_dof_stat = wilson_hilferty(statistic, dof)

                    ret_one_dof_stat[col] = one_dof_stat

                case _:
                    raise f"split_var role not handled in {self.__name__}"

        return ret_one_dof_stat

    def interaction_test(self, node) -> dict:
        """ Per the 2002 regression paper, calc one degree of freedom
        chi2 stats for interacting pairs """
        one_dof_stats = {}
        pairs = [*combinations(self.split_vars, r=2)]
        pvalues = {}
        one_dof_stats = {}
        residuals = self.df.loc[node.idx, self.tgt] - node.y_mean

        for pval_idx, (a, b) in enumerate(pairs):

            # case: a and b numeric
            if self.roles[a] in ['S', 'n'] and self.roles[b] in ['S', 'n']:
                chi_squared = np.zeros(shape=(2, 4))
                quadrants = list(product(['lt', 'gt'], ['lt', 'gt']))
                for idx, (ci, cj) in enumerate(quadrants):
                    if ci == 'lt':
                        left_bool_idx = self.df.loc[node.idx,
                                                    a] <= self.df.loc[node.idx, a].median()
                    else:
                        left_bool_idx = self.df.loc[node.idx,
                                                    a] > self.df.loc[node.idx, a].median()
                    if cj == 'lt':
                        right_bool_idx = self.df.loc[node.idx,
                                                     b] <= self.df.loc[node.idx, b].median()
                    else:
                        right_bool_idx = self.df.loc[node.idx,
                                                     b] > self.df.loc[node.idx, b].median()
                    chi_squared[0, idx] = (
                        residuals[left_bool_idx & right_bool_idx] <= 0).sum()
                    chi_squared[1, idx] = (
                        residuals[left_bool_idx & right_bool_idx] > 0).sum()
                chi_squared = remove_empty_cols(chi_squared)

            # case: a and b categoric
            elif self.roles[a] == self.roles[b] and self.roles[a] == 'c':
                alev = self.df.loc[node.idx, a].unique()
                blev = self.df.loc[node.idx, b].unique()
                cat_pairs = list(product(alev, blev))
                chi_squared = np.zeros(shape=(2, len(cat_pairs)))
                for idx, (ci, cj) in enumerate(cat_pairs):
                    chi_squared[0, idx] = (residuals[(self.df.loc[node.idx, a] == ci) & (
                        self.df.loc[node.idx, b] == cj)] < 0).sum()
                    chi_squared[1, idx] = (residuals[(self.df.loc[node.idx, a] == ci) & (
                        self.df.loc[node.idx, b] == cj)] >= 0).sum()
                chi_squared = remove_empty_cols(chi_squared)

            # case: one numeric and one categoric
            elif (self.roles[a] == 'c' and self.roles[b] in ['S', 'n']) or \
                    (self.roles[a] in ['S', 'n'] and self.roles[b] == 'c'):
                if self.roles[a] != 'c':
                    # ensure categorical variable is a
                    a, b = b, a
                alev = self.df.loc[node.idx, a].unique()
                chi_squared = np.zeros(shape=(2, 2 * len(alev)))
                groups = list(product(alev, ['lt', 'gt']))
                for idx, (ci, cj) in enumerate(groups):
                    if cj == 'lt':
                        bool_idx = self.df.loc[node.idx,
                                               b] <= self.df.loc[node.idx,
                                                                 b].median()
                    else:
                        bool_idx = (self.df.loc[node.idx, b]
                                    > self.df.loc[node.idx, b].median())
                    chi_squared[0, idx] = (residuals[(self.df.loc[node.idx, a] == ci) &
                                                     bool_idx] <= 0).sum()
                    chi_squared[1, idx] = (residuals[(self.df.loc[node.idx, a] == ci) &
                                                     bool_idx] > 0).sum()
                chi_squared = remove_empty_cols(chi_squared)

            else:
                assert False, "Unreachable line reached. Likely caused by unhandled variable role."

            res = chi2_contingency(chi_squared)
            dof, stat = res.dof, res.statistic
            one_dof_stat = wilson_hilferty(stat=stat, dof=dof)
            one_dof_stats[(a, b)] = one_dof_stat

        return one_dof_stats

    def _get_best_variable(self, node) -> str:
        """ Find best unbiased splitter among self.split_vars.
            1. Curvature tests
            2. Interaction test per the 2002 Regression paper. Note that the docs folder
            has a picture from the 2021 slideshow with another level of tests using linear
            discriminants.

            Algorithm 2. Choice between interacting pair of X variables.
            Suppose that a pair of variables is selected because their interaction test is
            the most signiﬁcant among all the curvature and interaction tests.
            1. If both variables are numerical-valued, the node is split in turn along the
            sample mean of each variable; for each split, the SSE for a constant model
            is obtained for each subnode; the variable yielding the split with the smaller
            total SSE is selected.
            2. Otherwise if at least one variable is categorical, the one with the smaller
            curvature p-value is selected.
            If a variable from a signiﬁcant interaction is selected to split a node, one
            strategy could be to require the other variable in the pair to split the immediate
            children nodes. This has the advantage of highlighting the interaction in the tree
            structure. On the other hand, by letting all the variables compete for splits at
            every node, it may be possible to obtain a shorter tree. The latter strategy is
            adopted for this reason.
        """

        if self.weight_var == list():
            node.y_mean = self.df.loc[node.idx, self.tgt].mean()
        else:
            # @NOTE: This is a little scaffolding for weigh var which is not implemented yet
            divisor = self.df.loc[node.idx, self.weight_var].sum()
            node.y_mean = (self.df.loc[node.idx,
                                       self.tgt] * self.df.loc[node.idx,
                                                               self.weight_var]).sum() / divisor

        curv_one_dof_stats = self.curvature_test(node)
        interaction_one_dof_stats = {}
        if self.interactions_on:
            interaction_one_dof_stats = self.interaction_test(node)

        curv_one_dof_stats = list(curv_one_dof_stats.items())
        interaction_one_dof_stats = list(interaction_one_dof_stats.items())
        interaction_pval = [(col, pvalue_for_one_dof(stat))
                            for col, stat in interaction_one_dof_stats]
        curv_pval = [(col, pvalue_for_one_dof(stat))
                     for col, stat in curv_one_dof_stats]

        # Bonferonni correction (uses statsmodels)

        # Tmp list of pvals to send to Bonferroni correction
        tmp_curv_pval_list = list(zip(*curv_pval))[1]
        curv_p_adj = multipletests(
            [*tmp_curv_pval_list], method='bonferroni')[1]
        curv_pval = [((first,), curv_p_adj[idx])
                     for idx, (first, _) in enumerate(curv_pval)]

        if self.interactions_on and len(interaction_pval) > 0:
            tmp_interact_pval_list = list(zip(*interaction_pval))[1]
            interact_p_adj = multipletests(
                [*tmp_interact_pval_list], method='bonferroni')[1]
            interaction_pval = [(first, interact_p_adj[idx])
                                for idx, (first, _) in enumerate(interaction_pval)]

        # Put the best corrected curvature and interaction pvalues at the top
        # of each list of tuples.

        # This approach is if instead of pre-sorting by chi-squared first.
        # We shuffle and sort by pval
        all_pval = []
        all_pval.extend(interaction_pval)
        all_pval.extend(curv_pval)

        # Use hashlib to generate a hash value
        dataframe_str = str(self.df.loc[node.idx].values)
        hash_object = hashlib.md5(dataframe_str.encode())
        rnd_seed = int(hash_object.hexdigest(), 16)

        # The adjusted pvals are shuffled deterministically
        # so that cases where all pvalues are the same (e.g. all 1.0)
        # do not make a biased selection.
        random.seed(rnd_seed)
        random.shuffle(all_pval)
        all_pval = sorted(all_pval, key=lambda x: (x[1], len(x[0])))

        top_var_is_singlet = True if len(all_pval[0][0]) == 1 else False

        all_curv_pval_are_same = len({pval for var, pval in curv_pval}) <= 1
        """
        if all_curv_pval_are_same is True and node.node_num == 1:
            print(f"random seed for pval == 1 {rnd_seed}")
        """

        best_var = None
        # Logic for interaction tests
        if not top_var_is_singlet:
            top_interact_pair = all_pval[0][0]
            # Select one of interacting pair
            # if one is categorical, split at the one with lower curvature pval
            # if both are numerical, split each at their mean
            role_a, role_b = self.roles[top_interact_pair[0]
                                        ], self.roles[top_interact_pair[1]]
            if role_a in ['n', 'S'] and role_b in ['n', 'S']:
                cut_a, cut_b = self.df.loc[node.idx, top_interact_pair[0]].mean(
                ), self.df.loc[node.idx, top_interact_pair[1]].mean()
                max_sse = None
                for col, cut in zip(top_interact_pair, [cut_a, cut_b]):
                    right_idx = (self.df.loc[node.idx, col] > cut).index
                    left_idx = self.df.drop(right_idx, axis=0).index
                    left_sse = ((self.df.loc[left_idx,
                                             self.tgt] - self.df.loc[left_idx,
                                                                     self.tgt].mean())**2).sum()
                    right_sse = ((self.df.loc[right_idx,
                                              self.tgt] - self.df.loc[right_idx,
                                                                      self.tgt].mean())**2).sum()
                    node_sse = ((self.df.loc[node.idx,
                                             self.tgt] - self.df.loc[node.idx,
                                                                     self.tgt].mean())**2).sum()
                    p = 1, len(left_idx) / \
                        len(node.idx), len(right_idx) / len(node.idx)
                    sse = p[0] * node_sse - p[1] * left_sse - p[2] * right_sse
                    if max_sse is None or sse > max_sse:
                        max_sse = sse
                        best_var = col
            else:
                logger.log(
                    logging.DEBUG, msg=f"UNTESTED CODE PATH HIT: interacting pair \
                            {top_interact_pair[0]},{top_interact_pair[1]} with one or two categoricals")
                curv_pval_a = [tup[1] for tup in curv_pval if tup[0]
                               [0] == top_interact_pair[0]][0]
                curv_pval_b = [tup[1] for tup in curv_pval if tup[0]
                               [0] == top_interact_pair[1]][0]

                if curv_pval_a == curv_pval_b:
                    # Make it random but deterministic if pvals match
                    random.seed(len(node.idx))
                    best_var = top_interact_pair[0] if random.randint(
                        0, 1) == 1 else top_interact_pair[1]
                else:
                    # Select variable with lowest curvature pvalue
                    best_var = top_interact_pair[np.argmin(
                        [curv_pval_a, curv_pval_b])]
        else:
            best_var = all_pval[0][0][0]

        if self.top_node_best_var is None:
            self.top_node_best_var = best_var

        if best_var is None:
            logger.log(
                level=logging.INFO,
                msg=f"file with bad best_var = {self.settings.overwrite_data_text}")
            pdb.set_trace()

        return best_var

    def fit(self):
        """ Build model from training data """
        self.node_list = [None] * 200  # all nodes of tree
        stack = [None] * 200     # nodes that need processed
        stack.clear()
        self.node_list.clear()
        self.top_node.node_num = self._get_next_node_num()
        stack.append(self.top_node)

        assert self.model_type == RegressionType.PIECEWISE_CONSTANT, "other models not implemented"

        # process nodes, adding new nodes as they are created
        while len(stack) > 0:
            curr = stack.pop(0)
            # get split variable and split point
            na_left = None
            split_var = self._get_best_variable(node=curr)
            if self.split_point_method == SplitPointMethod.Greedy:
                split_point, na_left = self._get_split_point_greedy(
                    node=curr, col=split_var)
            elif self.split_point_method == SplitPointMethod.Median:
                split_point, na_left = self._get_split_point_median(
                    node=curr, col=split_var)
            elif self.split_point_method == SplitPointMethod.Systematic:
                raise "not implemented"

            if False and curr.node_num == 14:
                pdb.set_trace()

            if split_point is None:
                curr.node_data = TerminalData(value=curr.y_mean)
                self.node_list.append(curr)
                continue

            assert isinstance(curr.idx, np.ndarray)
            _df = self.df.loc[curr.idx]
            predicate = None

            # create predicate (lambda) for splitting dataframe
            # can be printed with:
            #   from dill.source import getsource
            if isinstance(split_point, tuple):
                def predicate(
                    x, split_point=split_point): return x in split_point
            else:
                if na_left:
                    def predicate(
                        x, split_point=split_point): return x < split_point or np.isnan(x)
                else:
                    def predicate(
                        x, split_point=split_point): return x < split_point

            # Split dataframe
            # @NOTE can index a dataframe by a boolean but need to call .loc to index it with an index
            # logger.log(logging.DEBUG, msg = f"splitting node {curr.node_num} with split_var = {split_var} at split point = {split_point}")
            left = _df[_df[split_var].map(predicate)].index.values
            right = _df[~_df[split_var].map(predicate)].index.values
            assert left.shape[0] + right.shape[0] == curr.idx.shape[0]

            if left.shape[0] <= self.MIN_SAMPLES_LEAF or right.shape[0] <= self.MIN_SAMPLES_LEAF \
                    or curr.depth == self.MAX_DEPTH:
                # Based on early stopping, make curr node a leaf
                curr.node_data = TerminalData(value=curr.y_mean)
                self.node_list.append(curr)
                continue

            assert predicate is not None
            curr.node_data = InternalData(
                split_var=split_var,
                split_point=split_point,
                predicate=predicate,
                na_goes_left=na_left)

            # Split node
            left_node = Node(node_data=None, depth=curr.depth + 1, parent=curr,
                             indices=left, node_num=self._get_next_node_num())
            right_node = Node(
                node_data=None,
                depth=curr.depth + 1,
                parent=curr,
                indices=right,
                node_num=self._get_next_node_num())
            curr.left = left_node
            curr.right = right_node
            stack.append(left_node)
            stack.append(right_node)
            self.node_list.append(curr)

        # generate the tree text
        self._print_tree(self.top_node, depth=1)

    def _print_tree(self, node, depth):
        """ saves tree text to Model class.  recursively process the tree to match the reference output """
        spacer = "  "
        # base case terminal node
        if node.left is None and node.right is None:
            sn = (depth - 1) * spacer + \
                f"Node {node.node_num}: {self.tgt}-mean = {node.node_data.value:9f}"
            self.tree_text.append(sn)
            return

        # symbols for categorical versus numeric
        # print left branch
        split_point_numeric = isinstance(
            node.node_data.split_point, (int, float))
        left_sym = None
        right_sym = None
        if split_point_numeric:
            left_sym = '<='
            right_sym = '>'
        else:
            left_sym = '='
            right_sym = '/='

        # make split point printed output match ref output with quotes around
        # categories separated by commas
        printable_split_point = ""
        if isinstance(node.node_data.split_point, tuple):
            printable_split_point = ", ".join(
                ["\"" + str(i) + "\"" for idx, i in enumerate(node.node_data.split_point)])
        else:
            printable_split_point = node.node_data.split_point

        sl = (depth - 1) * spacer + \
            f"Node {node.node_num}: {node.node_data.split_var} {left_sym} {printable_split_point}"
        self.tree_text.append(sl)
        self._print_tree(node.left, depth + 1)
        # print right branch
        sr = (depth - 1) * spacer + \
            f"Node {node.node_num}: {node.node_data.split_var} {right_sym} {printable_split_point}"
        self.tree_text.append(sr)
        self._print_tree(node.right, depth + 1)

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
        predictions = pd.DataFrame(
            columns=["train", "node", "observed", "predicted"])

        for idx, row in self.df.iterrows():
            curr = self.top_node
            train = 'n'
            node = None
            observed = None
            predicted = None

            if idx in self.top_node.idx:
                train = 'y'

            # Get to leaf node
            while True:
                if isinstance(curr.node_data, InternalData):
                    feat = curr.node_data.split_var
                    predicate = curr.node_data.predicate
                    goes_left = predicate(row[feat])
                    if type(goes_left) != np.bool_ and type(goes_left) != bool:
                        pdb.set_trace()
                    if goes_left:
                        curr = curr.left
                    else:
                        curr = curr.right
                if isinstance(curr.node_data, TerminalData):
                    node = curr.node_num
                    observed = row[self.tgt]
                    predicted = self.df.loc[curr.idx, self.tgt].mean()
                    break

            df2 = pd.DataFrame({'train': train,
                                'node': node,
                                'observed': observed,
                                'predicted': predicted},
                               index=[idx])
            predictions = pd.concat([predictions, df2])

        if print_me:
            print()
            print(predictions.to_string(index=False))
        return predictions

#####################################################
# Helper functions
#####################################################


def wilson_hilferty(stat, dof) -> np.float64:
    """ approximately convert chi-squared with dof degrees of freedom to 1 degree of freedom """
    if dof == 1:
        return stat
    if dof == 0:
        return 0
    w1 = (math.sqrt(2 * stat) - math.sqrt(2 * dof - 1) + 1)**2 / 2
    temp = 7 / 9 + math.sqrt(dof) * ((stat / dof)**(1 / 3) - 1 + 2 / (9 * dof))
    w2 = max(0, temp ** 3)

    w = None
    if stat < (dof + 10 * math.sqrt(2 * dof)):
        w = w2
    elif stat >= (dof + 10 * math.sqrt(2 * dof) and w2 < stat):
        w = (w1 + w2) / 2
    else:
        w = w1
    return w


def pvalue_for_one_dof(stat):
    """ pvalue for a 1 dof chi squared statistic """
    return 1 - chi2.cdf(stat, 1)


def remove_empty_cols(t: np.ndarray):
    """ Remove empty columns, modifying t in place """
    cols = t.shape[1]
    empty_cols = [c for c in range(cols) if t[:, c].sum() == 0]
    return np.delete(t, empty_cols, 1)
