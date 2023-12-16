"""
- Parse input file
- Parse description file
- Parse data file
- Pre process (switching columns, dropping rows)
- Populate the settings object
"""

import os
import pdb
from enum import Enum
from collections import Counter
from typing import List
import numpy as np
import pandas as pd


class RegressionType(Enum):
    """Type of regression"""

    PIECEWISE_CONSTANT = 1
    LINEAR_PIECEWISE = 2  # unused


class SplitPointMethod(Enum):
    GREEDY = 1
    MEDIAN = 2  # unused
    SYSTEMATIC = 3  # unused


class Settings:
    """The settings object holds model parameters. This should probably just be a dictionary."""

    def __init__(
        self,
        data_dir,
        dsc_file,
        model=RegressionType.PIECEWISE_CONSTANT,
        out_file=None,
        max_depth=10,
        min_samples_leaf=6,
        input_file=None,
        overwrite_data_txt=None,
    ):
        self.datafile_name = None
        self.dsc_file = None
        self.datafile_start_line_idx = None
        self.col_data = pd.DataFrame()
        self.model = RegressionType.PIECEWISE_CONSTANT
        self.data_dir = data_dir
        self.model = model
        self.dsc_file = dsc_file
        self.max_depth = max_depth
        # The reference program has a formula to calculate this but we do not
        self.min_samples_leaf = min_samples_leaf
        self.input_file = input_file
        # GUIDE output file can be given to be used for tree comparisons during
        # testing
        self.out_file = out_file
        self.interactions_on = False
        # Use this argument filename in place of whatever is at top of .dsc
        # file
        self.overwrite_data_text = overwrite_data_txt
        self.missing_vals = []
        self.df: pd.DataFrame = pd.DataFrame()
        self.idx_active: np.ndarray = np.ndarray(shape=0)
        self.split_vars = []
        self.categorical_vars = []
        self.numeric_vars = []
        self.dependent_var = None
        self.weight_var = []
        self.roles = {}


def _vars_by_role(df, char: str) -> List[str]:
    """Return variables names that have particular GUIDE role in the analysis"""
    assert len(char) == 1, "variable roles are exactly one char long"
    return df[df["var_role"] == char].var_name.values.tolist()


def parse_input_file(settings: Settings):
    """Read model parameters from the GUIDE input file and save them
    to the settings object"""
    lines = []
    with open(settings.data_dir + settings.input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, l in enumerate(lines):
        l = l.strip()
        if "(max. no. split levels)" in l:
            # parse up to the first ( as a number
            en = l.find("(")
            settings.max_depth = int(l[0:en])

        if "(1=default min. node size" in l and l.startswith("2"):
            # parse lines[idx+1] to number
            _line = lines[idx + 1].strip()
            # either a bracket is on the next line or not
            # if not, we just strip the line and convert
            # it to a number
            en = _line.find("(")
            if en != -1:
                _line = _line[0:en]
            settings.min_samples_leaf = int(_line)

        if "(1=interaction tests, 2=skip them)" in l:
            settings.interactions_on = l.startswith("1")


def parse_description_file(settings: Settings):
    """The GUIDE .dsc file has the name of data file,
    the missing value labels, and the roles of each variable"""
    lines = []
    with open(settings.data_dir + settings.dsc_file, "r", encoding="utf-8") as f_desc:
        lines = f_desc.readlines()

    if settings.overwrite_data_text is None:
        settings.datafile_name = lines[0].rstrip("\n")
    else:
        settings.datafile_name = settings.overwrite_data_text

    settings.missing_vals = lines[1].split()
    settings.datafile_start_line_idx = lines[2].rstrip("\n")
    # Create col_data locally and save to settings at end of function
    col_data = pd.DataFrame(
        index=range(3, len(lines)), columns=["num", "var_name", "var_role"]
    )
    for i in np.arange(3, len(lines)):
        col_data.iloc[i - 3] = lines[i].split()
    return col_data


def parse_data(settings: Settings, show_output=True):
    """Parse the descr and data files. Modifies settings object so it
    can be used to build models.
    """
    out_str = []  # build this string up and optionally output at end
    # Parse Input file if present

    if settings.input_file is not None:
        assert os.path.exists(
            settings.data_dir + settings.input_file
        ), "input file not found"
        parse_input_file(settings)

    # Parse description file
    assert os.path.exists(settings.data_dir + settings.dsc_file), "desc file not found"
    col_data = parse_description_file(settings)

    out_str.append(">> Parsed description file <<")
    out_str.append("*" * 50)
    out_str.append(f"Datafile name            : {settings.datafile_name}")
    out_str.append(f"Missing value labels     : {settings.missing_vals}")
    out_str.append(f"Number of variables      : {len(col_data)}")
    role_counts = dict(Counter(col_data["var_role"].to_list()))
    out_str.append(f"Variable types           : {role_counts}")

    # Count number of m columns following a,c,n, and s
    # @NOTE: Unused right now
    m_role_association = {}
    m_variables = []
    if role_counts.get("m", 0) > 0:
        # process m role associations
        prev_role = None
        for idx, row in col_data.iterrows():
            curr_role = row.var_role
            if curr_role == "m":
                m_variables += [row.var_name]
                m_role_association[prev_role] = m_role_association.get(prev_role, 0) + 1
            prev_role = curr_role
        out_str.append(f"m variables associated   : {m_role_association}")
    out_str.append("*" * 50)

    # Parse data file (.txt)

    # @TODO: Add option to parse settings object from a dataframe instead of data.txt
    assert os.path.exists(
        settings.data_dir + settings.datafile_name
    ), "datafile not found"

    df = pd.read_csv(
        settings.data_dir + settings.datafile_name,
        delim_whitespace=True,
        na_values=settings.missing_vals,
        header=int(settings.datafile_start_line_idx) - 2,
    )

    assert (
        df.shape[1] == col_data.shape[0]
    ), "dsc and txt file have unequal column counts"

    dependent_var = _vars_by_role(col_data, "d")[0]
    out_str.append(f"Number of rows datafile  : {df.shape[0]}")
    out_str.append(f"Dependent variable       : {dependent_var}")
    # num_missing_in_d = df[dependent_var].isnull().sum()
    out_str.append(f"Model type is            : {str(settings.model)}")

    # Remove rows of dataframe with non-positive weight or
    # missing values in d, e, t, r or z variables)
    # @NOTE incomplete. Cases handled so far: missing value in d, nonpositive weight
    dependent_var_null_rows = df[df[dependent_var].isnull()]
    if len(dependent_var_null_rows) > 0:
        idx_missing_d = df[df[dependent_var].isnull()].index
    else:
        idx_missing_d = pd.Series([])
    out_str.append(f"Dropped missing d rows   : {len(idx_missing_d)}")

    weight_var = _vars_by_role(col_data, "w")
    if weight_var != []:
        weight_var = weight_var[0]
        out_str.append(f"Weight variable found    : {weight_var}")

    wgt_vars = _vars_by_role(col_data, "w")
    if len(wgt_vars) == 0:
        idx_zero_or_negative_weight = pd.Series([])
    else:
        idx_zero_or_negative_weight = df[
            df[_vars_by_role(col_data, "w")[0]] <= 0.0
        ].index
    if len(idx_zero_or_negative_weight) > 0:
        out_str.append(
            f"Dropped rows w/ weight < 0   : {len(idx_zero_or_negative_weight)}"
        )

    # GUIDE converts some roles to others depending on model parameters
    # Convert n to S and report how many
    n_var_idx = col_data[col_data.var_role == "n"].index
    col_data.loc[n_var_idx, "var_role"] = "S"
    out_str.append(f"Converted {n_var_idx.shape[0]} n variables to S variables")

    # If column type s, w, or d add min and max missing to the col_data table
    numeric_var_names = _vars_by_role(col_data, "S") + _vars_by_role(col_data, "w")

    # Gather variable statistics for printing out
    col_data["min"] = " "
    col_data["max"] = " "
    col_data["levels"] = " "
    col_data["missing"] = " "
    _missing_vals_in_noncategorical_flag = False

    for col in numeric_var_names:
        assert df[col].dtype in (float, int), "can't do min or max of non-numeric"
        idx = col_data[col_data.var_name == col].index[0]
        col_data.loc[idx, "min"] = df[col].min()
        col_data.loc[idx, "max"] = df[col].max()
        _missing_count = df[col].isnull().sum()
        # Check for missing values among non-categorical
        if _missing_count > 0:
            _missing_vals_in_noncategorical_flag = True
        col_data.loc[idx, "missing"] = _missing_count if _missing_count > 0 else " "

    if _missing_vals_in_noncategorical_flag:
        out_str.append("Missing values found in non-categorical variables.")

    categorical_vars = _vars_by_role(col_data, "c") + _vars_by_role(col_data, "m")

    # active indexes used for model fitting
    idx_active = (
        set(df.index.values.flatten())
        - set(idx_missing_d.values.flatten())
        - set(idx_zero_or_negative_weight.values.flatten())
    )
    idx_active = pd.Series(list(idx_active)).values

    # @TODO: Check if output of min, max, levels, etc matches the reference
    _missing_vals_in_categoricals_flag = False
    for col in categorical_vars:
        idx = col_data[col_data.var_name == col].index[0]
        _level_count = df.loc[idx_active, col].value_counts(dropna=True).index.shape[0]
        _missing_count = df.loc[idx_active, col].isnull().sum()
        col_data.loc[idx, "levels"] = _level_count
        if _missing_count > 0:
            col_data.loc[idx, "missing"] = _missing_count
            _missing_vals_in_categoricals_flag = True
        else:
            col_data.loc[idx, "missing"] = " "

    if _missing_vals_in_categoricals_flag:
        out_str.append("Missing values found in categorical variables.")
        out_str.append(" Separate categories will be created.")

    # Report some data statistics and model params to the user
    if weight_var != []:
        out_str.append("Weight variable range    : ")
        out_str.append(f"{df[weight_var].min():.4e}, {df[weight_var].max():.4e}")
    out_str.append("\n")
    out_str.append(f"{col_data[col_data.var_role != 'x']}")

    num_split_variables = len(_vars_by_role(col_data, "c")) + len(
        _vars_by_role(col_data, "S")
    )
    out_str.append(f"Number of split variables: {num_split_variables}")

    out_str.append(f"Max depth of tree     : {settings.max_depth}")
    out_str.append(f"Min samples per node  : {settings.min_samples_leaf}")
    out_str.append(f"Interaction tests done: {settings.interactions_on}")

    settings.col_data = col_data
    settings.df = df
    settings.idx_active = idx_active
    settings.split_vars = _vars_by_role(col_data, "S") + _vars_by_role(col_data, "c")
    settings.categorical_vars = _vars_by_role(col_data, "c")
    settings.numeric_vars = set(numeric_var_names) - set(_vars_by_role(col_data, "w"))
    settings.dependent_var = dependent_var
    settings.weight_var = weight_var
    settings.roles = {
        var: col_data[col_data["var_name"] == var].var_role.values[0]
        for var in settings.split_vars
    }

    if show_output:
        print("\n".join(out_str))
