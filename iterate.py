from pprint import pprint
from node import Node, Model
from parse import parse_data, Settings, RegressionType
import pdb

if __name__ == "__main__":
    settings = Settings(
#       data_dir="./tests/data-strikes2/",
        data_dir="./tests/data-strikes1/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
#       max_depth=7, min_samples_leaf=24)
        max_depth=4, min_samples_leaf=5)
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    model.print()
    model.predict_train_data(print_me = True)
