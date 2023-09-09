import pytest
import sys
import os
import pdb
import numpy as np
sys.path.append("..")
from parse import Settings, RegressionType, parse_data
from node import Model

@pytest.fixture
def ce_reg_tiny():
    settings = Settings(
        data_dir="./data-ce-reg-few-cols-single-tree/",
        dsc_file="ce2021reg.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=4)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    return settings, model


def test_ce_reg_tiny_split_vars(ce_reg_tiny):
    """ test split variables versus reference algo """
    _settings, _model = ce_reg_tiny
    pass

def test_ce_reg_tiny_split_points(ce_reg_tiny):
    """ test split points versus reference algo """
    _settings, _model = ce_reg_tiny
    pass

def test_ce_reg_tiny_mse(ce_reg_tiny):
    """ test mse versus reference algo """
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

