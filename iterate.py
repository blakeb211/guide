from pprint import pprint
from node import Node, Model
from parse import parse_data, Settings, RegressionType
import pdb

if __name__ == "__main__":
    settings = Settings(
        data_dir="./tests/data-strikes2/",
        dsc_file="data.dsc",
        model=RegressionType.LINEAR_PIECEWISE_CONSTANT,
        out_file="cons.out",
        input_file='cons.in')
    parse_data(settings=settings)
    model = Model(settings)
    model.fit()
    model.print()
    print("\n".join(model.tree_text))
    model.predict_train_data(print_me = False)
