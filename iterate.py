from pprint import pprint
from node import Node, Model
from parse import parse_data, Settings, RegressionType
import pdb

if __name__ == "__main__":
    settings = Settings(
        data_dir="./tests/data-tiniest2/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        max_depth=3, min_samples_leaf=2)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    model.print()
