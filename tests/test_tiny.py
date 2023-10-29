import pytest
import sys
import os
import pdb
import re
import numpy as np
sys.path.append("..")
from parse import Settings, RegressionType, parse_data
from node import Model

# These tests scrape data from the GUIDE output and compare 
# it to our output. Thus, GUIDE must be run first to run these.
"""
 Constant fitted to cases with missing values in regressor variables
 No nodewise interaction tests
 Split values for N and S variables based on exhaustive search
 Maximum number of split levels: 6
 Minimum node sample size: 2
 Top-ranked variables and 1-df chi-squared values at root node
      1  0.2230E+01   num1
      2  0.6667E+00   num2
      3  0.6667E+00   cat1

 Structure of final tree. Each terminal node is marked with a T.
  
 D-mean is mean of target in the node
 Cases fit give the number of cases used to fit node
 MSE is residual sum of squares divided by number of cases in node
        Node    Total    Cases Matrix    Node      Node          Split
        label   cases      fit  rank    D-mean     MSE           variable
           1        6        6     1  8.500E+00  3.470E+01   num1 
           2T       2        2     1  3.500E+00  2.450E+01   - 
           3T       4        4     1  1.100E+01  2.467E+01   - 
  
 Number of terminal nodes of final tree: 2
 Total number of nodes of final tree: 3

 Regression tree:

 Node 1: num1 <= 1.5000000
   Node 2: target-mean = 3.5000000
 Node 1: num1 > 1.5000000 or NA
   Node 3: target-mean = 11.000000

Regression tree:
 Node 1: cat1 = "3"
   Node 2: num1 <= 1.5000000
     Node 4: target-mean = 9.4156865
   Node 2: num1 > 1.5000000 or NA
     Node 5: target-mean = 8.1723145
 Node 1: cat1 /= "3"
   Node 3: cat1 = "1"
     Node 6: target-mean = 2.5528586
   Node 3: cat1 /= "1"
     Node 7: target-mean = 4.6206851
 
 ***************************************************************

"""

class RefData:
    pass

def parse_output_file_linear_piecewise(data_dir, fname):
    """ Parse key parts of GUIDE output so we can compare it """
    SECT1 = "Top-ranked variables and 1-df chi-squared values"
    SECT2 = "D-mean is mean of target in the node"
    SECT3 = "Regression tree:"
    with open(data_dir + fname) as f: 
        lines = f.readlines()
        top_ranked_root = []
        cases_per_node = {}
        mse_per_node = {}
        for idx, l in enumerate(lines):
            # Load top two ranked variables for splitting root
            if l.strip().startswith(SECT1):
                top_ranked_root.append(lines[idx+1][22:].strip())
                top_ranked_root.append(lines[idx+2][22:].strip())
            # Load number of cases fit by each node
            pattern = r'^\s*(\d+)T?\s+(\S+)\s+\S+\s+\S+\s+\S+\s+(\S+)\s+.*$'
            if l.strip().startswith(SECT2):
                index = idx + 5
                while True:
                    if (lines[index].strip() != ""):
                        raw_line = lines[index].strip()
                        match = re.match(pattern, raw_line)
                        if match:
                            node = int(match.group(1))
                            cases = match.group(2)
                            mse = match.group(3)
                            cases_per_node[node] = int(cases)
                            mse_per_node[node] = float(mse)
                        index = index + 1
                    else:
                        break
        print(top_ranked_root)
        print(cases_per_node)
        print(mse_per_node)
        assert len(cases_per_node) == len(mse_per_node)
        return top_ranked_root, cases_per_node, mse_per_node
        

@pytest.fixture(scope='session')
def tiny1():
    settings = Settings(
        data_dir="./data-tiniest/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=6)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    ref_data = parse_output_file_linear_piecewise(settings.data_dir,"cons.out")
    return settings, model, ref_data

@pytest.fixture(scope='session')
def tiny2():
    settings = Settings(
        data_dir="./data-tiniest2/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=6)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    ref_data = parse_output_file_linear_piecewise(settings.data_dir,"cons.out")
    return settings, model, ref_data



def test_top_2_split_vars(tiny1):
    """ tinyiest  - test split variables versus reference algo """
    _settings, _model, _ref_data  = tiny1
    pass

def test_first_split_point(tiny1):
    """ tinyiest  - test split points versus reference algo """
    _settings, _model, _ref_data  = tiny1
    pass

def test_second_split_point(tiny1):
    """ tiniest  - test split points versus reference algo """
    _settings, _model, _ref_data  = tiny1
    pass

def test_top_2_split_vars(tiny2):
    """ tiniest2  - test split variables versus reference algo """
    _settings, _model, _ref_data  = tiny2
    pass

def test_first_split_point(tiny2):
    """ tiniest2  - test split points versus reference algo """
    _settings, _model, _ref_data  = tiny2
    pass

def test_second_split_point(tiny2):
    """ tiniest2  - test split points versus reference algo """
    _settings, _model, _ref_data  = tiny2
    pass

    """ test mse versus reference algo """
    """
def test_ce_reg_tiny_mse(tiny1):
    _settings, _model = ce_reg_tiny
    test = _model.df.loc[:, _model.split_vars]

    predictions = _model.predict(test=test)
   
    guide_pred = None
    guide_pred_file = _settings.data_dir + "cons.node"
    assert os.path.exists(guide_pred_file)

    with open(guide_pred_file) as f:
        lines = f.readlines()
        guide_pred = []
        for line in lines:
            guide_pred.append(line[42:49])
        del guide_pred[0]

    # Compare guide_pred to predictions
    error = np.asarray(guide_pred,dtype=float) - predictions['pred']
    mse = np.mean(error**2)
    assert mse < 10E4
    """

