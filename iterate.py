from pprint import pprint
from node import Node, Model
from parse import parse_data, Settings, RegressionType

if __name__ == "__main__":
    settings = Settings(
        data_dir="./regression-lsr-CE-data/",
        dsc_file="ce2021reg.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
