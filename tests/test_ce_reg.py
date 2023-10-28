import pytest
import sys
import os
import pdb
import numpy as np
sys.path.append("..")
from parse import Settings, RegressionType, parse_data
from node import Model

# These tests scrape data from the GUIDE output and compare 
# it to our output. Thus, GUIDE must be run first to run these.
@pytest.fixture
def tiny1():
    settings = Settings(
        data_dir="./data-tiniest/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=6)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    return settings, model

@pytest.fixture
def tiny2():
    settings = Settings(
        data_dir="./data-tiniest2/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=6)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    return settings, model


def test_top_2_split_vars(tiny1):
    """ tinyiest  - test split variables versus reference algo """
    _settings, _model = tiny1
    pass

def test_first_split_point(tiny1):
    """ tinyiest  - test split points versus reference algo """
    _settings, _model = ce_reg_tiny
    pass

def test_second_split_point(tiny1):
    """ tiniest  - test split points versus reference algo """
    _settings, _model = ce_reg_tiny
    pass

def test_top_2_split_vars(tiny2):
    """ tiniest2  - test split variables versus reference algo """
    _settings, _model = tiny1
    pass

def test_first_split_point(tiny2):
    """ tiniest2  - test split points versus reference algo """
    _settings, _model = ce_reg_tiny
    pass

def test_second_split_point(tiny2):
    """ tiniest2  - test split points versus reference algo """
    _settings, _model = ce_reg_tiny
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

