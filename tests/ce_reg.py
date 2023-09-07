import pytest

from ..parse import Settings, RegressionType
from ..node import Model

@pytest.fixture
def ce_reg_no_M_cols:
    settings = Settings(
        data_dir="./regression-lsr-CE-data/",
        dsc_file="ce2021reg.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    return model
    


def test_ce_single_tree_no_prune_predictions(ce_reg_no_M_cols):
    _model = ce_reg_no_M_cols
    test = _model.df.loc[:, _model.split_vars]

    predictions = _model.predict(test=test)
   
    guide_pred = None
    with open("regression-lsr-CE-data/cons.predicted") as f:
        lines = f.readlines()
        guide_pred = []
        for line in lines:
            guide_pred.append(line[42:49])
        del guide_pred[0]

    # compare guide_pred to predictions

def test_ce_single_tree_no_prune_model_params()
    pass





    
