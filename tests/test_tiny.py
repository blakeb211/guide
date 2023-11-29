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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Test Logger')

# helpers
def represents_num(s):
    """ test if string can be cast to a float """
    try: 
        float(s)
    except ValueError:
        return False
    else:
        return True

# These tests scrape data from the GUIDE output and compare 

def parse_output_file_for_tree_text(data_dir, fname):
    """ Parse parts of GUIDE output so we can compare it.
        May strip this down to separate functions if they 
        are needed to test against.
    """
    SECT3 = "Regression tree:"
    SECT3END = "***********"
    with open(data_dir + fname) as f: 
        lines = f.readlines()
        tree_text = ""
        for idx, l in enumerate(lines):
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

        return tree_text
        
@pytest.fixture(scope='session')
def tiny2(): 
    settings = Settings(
        data_dir="./data-tiniest2/",
        dsc_file="data.dsc",
        out_file="cons.out",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=3, min_samples_leaf=2)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    return settings, model, predictions

@pytest.fixture(scope='session')
def strikes1(): 
    settings = Settings(
        data_dir="./data-strikes1/",
        dsc_file="data.dsc",
        out_file="cons.out",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        input_file="cons.in")
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    return settings, model, predictions

@pytest.fixture(scope='session')
def strikes1_deep(): 
    settings = Settings(
        data_dir="./data-strikes1-deep/",
        dsc_file="data.dsc",
        out_file="cons.out",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        input_file="cons.in")
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    return settings, model, predictions


@pytest.fixture(scope='session')
def strikes2(): 
    settings = Settings(
        data_dir="./data-strikes2/",
        dsc_file="data.dsc",
        out_file="cons.out",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        input_file="cons.in")
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    predictions = model.predict_train_data()
    return settings, model, predictions


def compare_predicted_vals(ref, this_prog):
    """ compare program output to the .node file (train data run through the model) """
    cutoff = 1E-3
    titles_match =  this_prog.columns == ref.columns 
    train_y_or_n_matches =  this_prog.train == ref.train 
    observed_differences =  np.abs(this_prog.observed - ref.observed)
    prediction_differences = np.abs(this_prog.predicted - ref.predicted)

    logger.log(logging.INFO, msg = f"num cases match?               {this_prog.shape[0] == ref.shape[0]} this_prog, ref = {this_prog.shape[0]},{ref.shape[0]}")
    logger.log(logging.INFO, msg = f"column titles match?           {titles_match.all()}")
    logger.log(logging.INFO, msg = f"train y or n matches?          {train_y_or_n_matches.all()}")
    logger.log(logging.INFO, msg = f"observed difference max        {observed_differences.max():.2g}")
    logger.log(logging.INFO, msg = f"prediction difference max      {prediction_differences.max():.2g}")
     
    assert titles_match.all() and train_y_or_n_matches.all() and (observed_differences < cutoff).all() and (prediction_differences < cutoff).all()

def compare_trees(ref_tree, tree_text):
    """ compare split variable and split points between two trees """
    # r = reference
    # p = our program
    cutoff = 1E-4

    regex = r"""Node\s(\d*):\s(\S+)\s([<>=//]+)(.*)"""
    first_point_of_diff = None 

    for lr, lp in zip(ref_tree, tree_text):
        rtup = re.findall(regex, lr)[0] # returns list of tup so we take index 0
        ptup = re.findall(regex, lp)[0]
        if rtup[1] != ptup[1]:
            first_point_of_diff = f"split var at node {ptup[0]}"
            break
        if rtup[2] != ptup[2]:
            first_point_of_diff = f"splitting sign at node {ptup[0]}"
            break

        # @TODO Trim off the "or NA" part of the Guide split points for now because
        #       we have not studied the missing value behavior yet or implemented any of it.
        rtup = rtup[0], rtup[1], rtup[2], rtup[3].rstrip(' or NA')

        if ptup[1] == 'cat1' and int(ptup[0]) == 8:
            pdb.set_trace()
        # split point can be numeric or a list of quoted values
        split_point_same = False
        if represents_num(rtup[3]):
            # if one split point is num, both should be numbers since we have the same split_var,
            # but if something weird happened we want to know 
            assert(represents_num(ptup[3]))
            split_point_same = np.isclose(float(ptup[3]),float(rtup[3]))
        else:
            # handle comparing categories
            split_point_same = sorted(rtup[3].split(',')) == sorted(ptup[3].split(','))
            
        if not split_point_same:
            first_point_of_diff = f"split point at node {ptup[0]}"
            break

    logger.log(logging.INFO, msg = f"tree point of difference       {first_point_of_diff}")
    if first_point_of_diff != None:
        logger.log(logging.INFO, msg = f"************ {lp} vs {lr}")
    assert first_point_of_diff == None



def test_tiny2(tiny2):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            all numeric variables 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = tiny2
    
    ref_tree = parse_output_file_for_tree_text(data_dir=_settings.data_dir, fname=_settings.out_file) 
    compare_trees(ref_tree, _model.tree_text)

    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)
    compare_predicted_vals(reference, _predictions)
    

def test_strikes1(strikes1):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            cat and numeric vars 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = strikes1
    
    ref_tree = parse_output_file_for_tree_text(data_dir=_settings.data_dir, fname=_settings.out_file) 
    compare_trees(ref_tree, _model.tree_text)

    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)
    compare_predicted_vals(reference, _predictions)

def test_strikes1_deep(strikes1_deep):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            cat and numeric vars 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = strikes1_deep
    
    ref_tree = parse_output_file_for_tree_text(data_dir=_settings.data_dir, fname=_settings.out_file) 
    compare_trees(ref_tree, _model.tree_text)

    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)
    compare_predicted_vals(reference, _predictions)
    

def test_strikes2(strikes2):
    """ Compare predictions of fitted model on the training data to
    reference software output. 
    Case:   piecewise constant
            no weight var
            cat and numeric vars 
            no missing values
            no interaction test
    """
    _settings, _model, _predictions = strikes2

    ref_tree = parse_output_file_for_tree_text(data_dir=_settings.data_dir, fname=_settings.out_file) 
    compare_trees(ref_tree, _model.tree_text)

    reference = pd.read_csv(_settings.data_dir + "data.node", delim_whitespace=True)
    compare_predicted_vals(reference, _predictions)
