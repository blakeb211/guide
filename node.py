""" First attempt at reproducing GUIDE output for
single tree regression test case
- parse description file
- parse data file
- generate predictions that are not that good to build out framework
"""
import sys
import os
import pdb
from enum import Enum
import numpy as np
import pandas as pd


class PredictionType(Enum):
    """ Classification or regression """
    CLASS = 1
    REG = 2


class RegressionType(Enum):
    """ Type of regression """
    LINEAR_PIECEWISE_CONSTANT = 1
    QUANTILE = 2
    POISSON = 3


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


# Globals
datafile_name = None
missing_vals = list()
datafile_start_line_idx = None
col_data = pd.DataFrame()
data_dir = "./regression-lsr-CE-data/"
model = RegressionType.LINEAR_PIECEWISE_CONSTANT

def _variables_by_role(char: str):
    """ Return variables names that have particular GUIDE role in the analysis """
    assert len(char) == 1, "variable roles are exactly one char long"
    return col_data[col_data['var_role'] == char].var_name.values


def parse_data(description_file: str):
    """ Parse the descr and data files """
    # Parse description file first
    with open(description_file, "r") as f:
        lines = f.readlines()
        global datafile_name
        datafile_name = lines[0].rstrip('\n')
        global missing_vals
        missing_vals = lines[1].split()
        global datafile_start_line_idx
        datafile_start_line_idx = lines[2].rstrip('\n')
        global col_data
        col_data = pd.DataFrame(
            index=range(
                3,
                len(lines)),
            columns=[
                "num",
                "var_name",
                "var_role"])
        for i in np.arange(3, len(lines)):
            col_data.iloc[i - 3] = lines[i].split()
    assert os.path.exists(
        data_dir + datafile_name), f"{row_data_file} not found"
    print(">> Parsed description file <<")
    print('*' * 50)
    print(f"Datafile name            : {datafile_name}")
    print(f"Missing value labels     : {missing_vals}")
    print(f"Number of variables      : {len(col_data)}")
    print(f"Variable types           : {col_data['var_role'].unique()}")
    # @TODO: Count number of m columns following a,c,n, and s
    print('*' * 50)

    # Parse data file (.txt)
    """
                                                      #Codes/
                                                      Levels/
      Column  Name            Minimum      Maximum    Periods   #Missing


      Total  #cases w/   #missing
     #cases    miss. D  ord. vals   #X-var   #N-var   #F-var   #S-var
       3965       1478       3965        1        0        0      384
     #P-var   #M-var   #B-var   #C-var   #I-var
          0      116        0       47        0
     Weight variable FINLWT21 in column: 31
     Number of cases used for training: 2487
     Number of split variables: 431
     Number of cases excluded due to 0 W or missing D variable: 1478
    """
    df = pd.read_csv(data_dir + datafile_name, delimiter=" ")  # MS-DOS CSV
    assert df.shape[1] == col_data.shape[0], "dsc and txt file have unequal column counts"
    dependent_var = col_data[col_data['var_role'] == 'd'].var_name.values
    print(f"Number of rows datafile  : {df.shape[0]}")
    print(f"Dependent variable       : {_variables_by_role('d')[0]}")
    if df[dependent_var].isnull().sum().values[0] > 0:
        print("Missing values found in d variable")
    print(f"Model type is {str(model)}")

    # Create variable data printout for min, max, number categories, missing
    col_data['min'] = ' '
    col_data['max'] = ' '
    col_data['levels'] = ' '
    col_data['missing'] = ' '
    # Convert n to S and report how many
    n_var_idx = col_data[col_data.var_role == 'n'].index
    col_data.loc[n_var_idx, 'var_role'] = 'S'
    print(f"Converted {n_var_idx.shape[0]} n variables to S variables")
    # @TODO: If column type s, w, or d add min and max to the col_data table

    print(col_data)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(f"Arguments given:", sys.argv)
    # Description file
    dsc_file = "ce2021reg.dsc"
    assert os.path.exists(data_dir + dsc_file), f"{desc_file} not found"
    parse_data(data_dir + dsc_file)
