import pytest
import sys
import os
import pdb
import re
import logging
import numpy as np
import pandas as pd
sys.path.append("..")
from parse import Settings, RegressionType, parse_data
from node import Model

logging.basicConfig(level=logging.CRITICAL)
logger = logging.getLogger('Test Logger')

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
    logger.log(logging.CRITICAL, msg = f"fixture loaded")
    return settings, model, predictions

@pytest.fixture(scope='session')
def strikes1(): 
    settings = Settings(
        data_dir="./data-strikes1/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=2, min_samples_leaf=5)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    logger.log(logging.CRITICAL, msg = f"fixture loaded")
    return settings, model, predictions

@pytest.fixture(scope='session')
def strikes1_deep(): 
    settings = Settings(
        data_dir="./data-strikes1-deep/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=4, min_samples_leaf=5)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    logger.log(logging.CRITICAL, msg = f"fixture loaded")
    return settings, model, predictions

def test_node_file_predictions_for_tiny2(tiny2):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            all numeric variables 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = tiny2
    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)

    titles_match =  _predictions.columns == reference.columns 
    train_y_or_n_matches =  _predictions.train == reference.train 
    nodes_that_cases_landed_in_match =  _predictions.node == reference.node 
    observed_differences =  np.abs(_predictions.observed - reference.observed)
    prediction_differences = np.abs(_predictions.predicted - reference.predicted)

    num_cases = _predictions.shape[0]
    logger.log(logging.CRITICAL, msg = f"num cases = {num_cases} max pred diff = {prediction_differences.max():.2g}")
    
    assert titles_match.all()
    assert train_y_or_n_matches.all() 
    assert nodes_that_cases_landed_in_match.all()
    assert (observed_differences < 1E-3).all()
    assert (prediction_differences < 1E-3).all()

def test_node_file_predictions_for_strikes1(strikes1):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            categoric and numeric variables 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = strikes1 
    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)

    titles_match =  _predictions.columns == reference.columns 
    train_y_or_n_matches =  _predictions.train == reference.train 
    nodes_that_cases_landed_in_match =  _predictions.node == reference.node 
    observed_differences =  np.abs(_predictions.observed - reference.observed)
    prediction_differences = np.abs(_predictions.predicted - reference.predicted)

    num_cases = _predictions.shape[0]
    logger.log(logging.CRITICAL, msg = f"num cases = {num_cases} max pred diff = {prediction_differences.max():.2g}")
    
    assert titles_match.all()
    assert train_y_or_n_matches.all() 
    assert nodes_that_cases_landed_in_match.all()
    assert (observed_differences < 1E-3).all()
    assert (prediction_differences < 1E-3).all()

def test_node_file_predictions_for_strikes1_deep(strikes1_deep):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            categoric and numeric variables 
            no missing values
            no interaction test
            *slightly deeper tree
    """
    _settings, _model, _predictions = strikes1_deep
    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)

    titles_match =  _predictions.columns == reference.columns 
    train_y_or_n_matches =  _predictions.train == reference.train 
    nodes_that_cases_landed_in_match =  _predictions.node == reference.node 
    observed_differences =  np.abs(_predictions.observed - reference.observed)
    prediction_differences = np.abs(_predictions.predicted - reference.predicted)

    num_cases = _predictions.shape[0]
    logger.log(logging.CRITICAL, msg = f"num cases = {num_cases} max pred diff = {prediction_differences.max():.2g}")
    
    assert titles_match.all(), "titles did not match"
    assert train_y_or_n_matches.all(), "train y or no did not match"
    print(f"node numbers match = {nodes_that_cases_landed_in_match.all()}")
    assert (observed_differences < 1E-3).all(), "observed diffs did not match"
    assert (prediction_differences < 1E-3).all(), "prediction diffs did not match"
