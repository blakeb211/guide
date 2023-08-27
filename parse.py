""" First attempt at reproducing GUIDE output for
single tree regression test case
- parse description file
- parse data file
- Pre process (switching columns, dropping rows)

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

import sys
import os
import pdb
from enum import Enum
import numpy as np
import pandas as pd
from collections import Counter


class RegressionType(Enum):
    """ Type of regression """
    LINEAR_PIECEWISE_CONSTANT = 1
    QUANTILE = 2
    POISSON = 3


# Globals
class Settings():
    df = None
    datafile_name = None
    dsc_file = None
    missing_vals = list()
    datafile_start_line_idx = None
    col_data = pd.DataFrame()
    data_dir = "./regression-lsr-CE-data/"
    model = RegressionType.LINEAR_PIECEWISE_CONSTANT
    weight_var = None
    numeric_vars = None
    categorical_vars = None
    dependent_var = None
    # Variables to use for splitting and/or fitting are 
    # determined during parsing just like every other
    # model parameter.
    split_vars = None
    fit_vars = None
    m_variables = None
    MAX_DEPTH = 12
    MIN_SAMPLES_NODE = 20

    def __init__(self, data_dir, dsc_file, model):
        self.data_dir = data_dir
        self.model = model
        self.dsc_file = dsc_file
        assert os.path.exists(self.data_dir +
                              self.dsc_file), f"{self.dsc_file} not found"


def _variables_by_role(df, char: str) -> list():
    """ Return variables names that have particular GUIDE role in the analysis """
    assert len(char) == 1, "variable roles are exactly one char long"
    return df[df['var_role'] == char].var_name.values.tolist()


def parse_data(settings : Settings):
    """ Parse the descr and data files. Modifies settings so it
    can be used to build models.
    """
    # Parse description file first
    # @NOTE: Could simplify by making a local dict vars_by_role
    description_file = settings.data_dir + settings.dsc_file
    with open(description_file, "r") as f:
        lines = f.readlines()
        settings.datafile_name = lines[0].rstrip('\n')
        settings.missing_vals = lines[1].split()
        settings.datafile_start_line_idx = lines[2].rstrip('\n')
        # Create col_data locally and save to settings at end of function
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
        settings.data_dir + settings.datafile_name), f"{row_data_file} not found"
    print(">> Parsed description file <<")
    print('*' * 50)
    print(f"Datafile name            : {settings.datafile_name}")
    print(f"Missing value labels     : {settings.missing_vals}")
    print(f"Number of variables      : {len(col_data)}")
    role_counts = dict(Counter(col_data['var_role'].to_list()))
    print(f"Variable types           : {role_counts}")
    # Count number of m columns following a,c,n, and s
    m_role_association = {}
    m_variables = []
    if role_counts.get('m', 0) > 0:
        # process m role associations
        prev_role = None
        for (idx, row) in col_data.iterrows():
            curr_role = row.var_role
            if curr_role == 'm':
                m_variables += [row.var_name]
                m_role_association[prev_role] = m_role_association.get(
                    prev_role, 0) + 1
            prev_role = curr_role
        print(f"m variables associated   : {m_role_association}")
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
    df = pd.read_csv(
        settings.data_dir + settings.datafile_name,
        delimiter=" ",
        na_values=settings.missing_vals)  # MS-DOS CSV
    assert df.shape[1] == col_data.shape[0], "dsc and txt file have unequal column counts"
    dependent_var = _variables_by_role(col_data, 'd')[0]
    print(f"Number of rows datafile  : {df.shape[0]}")
    print(f"Dependent variable       : {dependent_var}")
    num_missing_in_d = df[dependent_var].isnull().sum()
    print(f"Model type is            : {str(settings.model)}")

    # Remove rows of dataframe with non-positive weight or missing values in d, e, t, r or z variables)
    # @NOTE incomplete. Cases handled so far: missing value in d, nonpositive weight
    idx_missing_d = df[df[dependent_var].isnull() == True].index
    df.drop(idx_missing_d, inplace=True)
    print(f"Dropped missing d rows   : {len(idx_missing_d)}")

    weight_var = _variables_by_role(col_data, 'w')
    if weight_var != list():
        weight_var = weight_var[0]
        print(f"Weight variable found    : {weight_var}")
    idx_zero_or_negative_weight = df[df[_variables_by_role(col_data, 'w')[
        0]] <= 0.0].index
    df.drop(idx_zero_or_negative_weight, inplace=True)
    if len(idx_zero_or_negative_weight) > 0:
        print(
            f"Dropped rows w/ weight < 0   : {len(idx_zero_or_negative_weight)}")

    # Create variable data printout for min, max, number categories, missing
    col_data['min'] = ' '
    col_data['max'] = ' '
    col_data['levels'] = ' '
    col_data['missing'] = ' '

    # Convert n to S and report how many
    n_var_idx = col_data[col_data.var_role == 'n'].index
    col_data.loc[n_var_idx, 'var_role'] = 'S'
    print(f"Converted {n_var_idx.shape[0]} n variables to S variables")

   # If column type s, w, or d add min and max missing to the col_data table
    numeric_var_names = _variables_by_role(col_data, 'S') + _variables_by_role(col_data, 'w')

    _missing_vals_in_noncategorical_flag = False
    for col in numeric_var_names:
        assert df[col].dtype == np.float64 or df[col].dtype == np.int64, "calculating min,max,missing of non-numeric column"
        idx = col_data[col_data.var_name == col].index[0]
        col_data.loc[idx, 'min'] = df[col].min()
        col_data.loc[idx, 'max'] = df[col].max()
        _missing_count = df[col].isnull().sum()
        # Check for missing values among non-categorical
        if _missing_count > 0:
            _missing_vals_in_noncategorical_flag = True
        col_data.loc[idx,
                     'missing'] = _missing_count if _missing_count > 0 else ' '

    if _missing_vals_in_noncategorical_flag == True:
        print(f"Missing values found in non-categorical variables.")

    categorical_vars = _variables_by_role(col_data, 'c') + _variables_by_role(col_data, 'm')

    # Tally levels for categoricals
    # tell user that separate categories will be created
    # @TODO: Match levels output for 'm' and 's' variables with the reference.
    # @NOTE: Levels match reference for 'c' roles but not other roles,
    # but the 'm' columns do not appear in the split vars of the final tree,
    # so at minimum we can skip for now.

    _missing_vals_in_categoricals_flag = False
    for col in categorical_vars:
        idx = col_data[col_data.var_name == col].index[0]
        _level_count = df[col].value_counts(dropna=True).index.shape[0]
        _missing_count = df[col].isnull().sum()
        col_data.loc[idx, 'levels'] = _level_count
        if _missing_count > 0:
            col_data.loc[idx, 'missing'] = _missing_count
            _missing_vals_in_categoricals_flag = True
        else:
            col_data.loc[idx, 'missing'] = ' '

    if _missing_vals_in_categoricals_flag == True:
        print(f"Missing values found in categorical variables. Separate categories will be created.")

    # Report min, max of weights,
    print(
        f"Weight variable range    : {df[weight_var].min():.4e}, {df[weight_var].max():.4e}")
    print()
    print(col_data)
    print(
        f"Number of split variables: {len(_variables_by_role(col_data, 'c')) + len(_variables_by_role(col_data, 'S'))}")
    x_vars = _variables_by_role(col_data, 'x')
    for excluded_column in x_vars:
        del df[excluded_column]
    settings.col_data = col_data
    settings.df = df
    settings.split_vars = _variables_by_role(col_data, 'S') + _variables_by_role(col_data, 'c')
    settings.fit_vars = _variables_by_role(col_data, 'n')
    settings.categorical_vars = _variables_by_role(col_data, 'c') 
    settings.numeric_vars = numeric_var_names
    settings.dependent_var = dependent_var
