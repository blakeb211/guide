"""
Change a few parameters and run the model.
Used for iterating on the project.
"""
import pdb
from node import Model
from parse import Settings, RegressionType
import pandas as pd

if __name__ == "__main__":
    settings = Settings(
        data_dir="./tests/data-tiniest2/",
        dsc_file="data.dsc",
        model=RegressionType.PIECEWISE_CONSTANT,
        out_file="cons.out",
        input_file="cons.in",
    )
    model = Model(settings)
    model.fit()
    print("\n".join(model.tree_text))
    # model.predict_train_data(print_me=True)
