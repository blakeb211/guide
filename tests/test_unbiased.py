""" Test Unbiased variable selection for This Program, and CART.
See the 2002 regression paper, Table 4, pg 367.
1-  generate fresh data for     C5, C10, U, T, W, and Z
2-  write data.txt for independent, weakly dependent, strongly dependent cases
3-  fit each dataset with a piecewise constant model
4-  tally variable selected at root node
5-  repeat 1000 times = 3000 total fits
6-  compare the counts divided by 1000 (frequency) to 0.2, 
    which is the complete unbiased case (5 variables 1000 sims)
"""
import sys
import pdb
import pathos.pools as pp
import re
import logging
import numpy as np
import pandas as pd
sys.path.append("..")
from node import Model
from parse import Settings, RegressionType


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test Logger')


def test_unbiased_selection_no_interaction_tests():
    data_dir = "./data-unbiased-selection/"

    def fit_and_tally(fname):
        settings = Settings(
            data_dir=data_dir,
            dsc_file="data.dsc",
            model=RegressionType.PIECEWISE_CONSTANT,
            input_file="cons.in",
            overwrite_data_txt=fname)
        model = Model(settings, show_parse_output=False)
        model.fit()
        return model.top_node_best_var

    def run_simulation_iteration(i):
        count = 1000  # instances and test repetitions
        np.random.seed(seed=i)
        C5 = np.random.randint(0, 5, size=count)
        C10 = np.random.randint(0, 10, size=count)
        U = np.random.uniform(low=0.0, high=1.0, size=count)
        T = np.random.choice(np.asarray([-1, 1, -3, 3]), size=count)
        W = np.random.exponential(scale=1.0 / 1.0, size=count)
        Z = np.random.standard_normal(size=count)
        Y = np.random.standard_normal(size=count)  # all cases

        X1 = T  # all cases
        X2 = W  # all cases
        X3_1 = Z
        X3_2 = T + W + Z
        X3_3 = W + 0.1 * Z
        X4_1 = C5
        X4_2 = np.floor(U * C10 / 2) + 1
        X4_3 = np.floor(U * C10 / 2) + 1
        X5 = C10  # all cases

        col_list = ["X1", "X2", "X3", "X4", "X5", "Y"]
        indep = pd.DataFrame(
            np.column_stack(
                (X1, X2, X3_1, X4_1, X5, Y)), columns=col_list)
        weak = pd.DataFrame(
            np.column_stack(
                (X1, X2, X3_2, X4_2, X5, Y)), columns=col_list)
        strong = pd.DataFrame(
            np.column_stack(
                (X1, X2, X3_3, X4_3, X5, Y)), columns=col_list)

        with open(data_dir + f"data-indep{i}.txt", "w", encoding="utf-8") as f:
            f.write(indep.to_string(col_space=10, index=False))
        with open(data_dir + f"data-weak{i}.txt", "w", encoding="utf-8") as f:
            f.write(weak.to_string(col_space=10, index=False))
        with open(data_dir + f"data-strong{i}.txt", "w", encoding="utf-8") as f:
            f.write(strong.to_string(col_space=10, index=False))

        split_var_indep = fit_and_tally(f"data-indep{i}.txt")
        split_var_weak = fit_and_tally(f"data-weak{i}.txt")
        split_var_strong = fit_and_tally(f"data-strong{i}.txt")
        return split_var_indep, split_var_weak, split_var_strong

    pool = pp.ProcessPool(16)
    SIM_COUNT = 1000
    results = pool.map(run_simulation_iteration, range(SIM_COUNT))

    # collect results
    big_dict_indep = {}
    big_dict_weak = {}
    big_dict_strong = {}

    for (ind_split_var, weak_split_var, strong_split_var) in results:
        big_dict_indep[ind_split_var] = big_dict_indep.get(
            ind_split_var, 0) + 1
        big_dict_weak[weak_split_var] = big_dict_weak.get(
            weak_split_var, 0) + 1
        big_dict_strong[strong_split_var] = big_dict_strong.get(
            strong_split_var, 0) + 1

    results_df = pd.DataFrame([big_dict_indep, big_dict_weak, big_dict_strong], index=[
                              "indep", "weak", "strong"]).transpose()
    
    assert results_df.sum().sum() == SIM_COUNT * 3
    results_df = results_df / SIM_COUNT
    results_df = results_df.sort_index()

    diffs_from_mean = np.abs(results_df - 0.2)
    three_std_err = 0.05
    # These are not the exact criteria that are in Table 4 of the 2002 paper (see docs folder)
    # In the paper, all values are within .05 of the completely unbiased value of 0.2

    outliers = 0
    outliers += (diffs_from_mean > three_std_err).sum().sum()

    logger.log(logging.INFO, "\n%s" % results_df.to_string())
    logger.log(logging.INFO, "outliers from unbiased (0.2) %s" % outliers)

    assert outliers < 2
