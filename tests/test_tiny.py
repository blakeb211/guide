import pytest
import sys
import os
import pdb
import re
import numpy as np
import pandas as pd
sys.path.append("..")
from parse import Settings, RegressionType, parse_data
from node import Model

# These tests scrape data from the GUIDE output and compare 

def parse_output_file_linear_constant(data_dir, fname):
    """ Parse parts of GUIDE output so we can compare it.
        May strip this down to separate functions if they 
        are needed to test against.
    """
    SECT1 = "Top-ranked variables and 1-df chi-squared values"
    SECT2 = "D-mean is mean of target in the node"
    SECT3 = "Regression tree:"
    SECT3END = "***********"
    with open(data_dir + fname) as f: 
        lines = f.readlines()
        top_ranked_root = []
        cases_per_node = {}
        mse_per_node = {}
        tree_text = ""
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
            if l.strip().startswith(SECT3):
                index = idx
                end_index = -1
                start_found = False
                while (True):
                    if start_found == False and lines[index].strip().startswith("Node 1"):
                        idx = index
                        start_found = True
                    if lines[index].strip().startswith(SECT3END):
                        end_index = index 
                        break
                    index = index + 1
                # grab the tree text into a list
                tree_text = lines[idx:end_index-1]
                break

        print(top_ranked_root)
        print(cases_per_node)
        print(mse_per_node)
        print(tree_text)
        assert len(cases_per_node) == len(mse_per_node)
        return top_ranked_root, cases_per_node, mse_per_node, tree_text
        
@pytest.fixture(scope='session')
def tiny2(): 
    settings = Settings(
        data_dir="./data-tiniest2/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=3, min_samples_leaf=2)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    return settings, model, predictions

def test_node_file_predictions(tiny2):
    """ Compare predictions of fitted model on the training data,
    including the node number they were made on. This is 
    essentially a downstream integration test of the model versus the reference. """
    _settings, _model, _predictions = tiny2
    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)
    titles_match =  _predictions.columns == reference.columns 
    assert titles_match.all()
    nodes_that_cases_landed_in_match =  _predictions.node == reference.node 
    assert nodes_that_cases_landed_in_match.all()
    observed_differences =  _predictions.observed - reference.observed 
    prediction_differences =  _predictions.predicted - reference.predicted 
    assert (observed_differences < 1E-3).all()
    assert (prediction_differences < 1E-3).all()
