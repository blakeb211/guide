"""
Change a few parameters and run the model.
Used for iterating on the project.
"""
import pdb
from node import Model
from parse import Settings, RegressionType

if __name__ == "__main__":
    settings = Settings(
        data_dir="./tests/data-unbiased-selection/",
        dsc_file="data.dsc",
        model=RegressionType.PIECEWISE_CONSTANT,
        out_file="cons.out",
        input_file="cons.in",
        overwrite_data_txt="data-weak22.txt")
    model = Model(settings)
    model.fit()
    model.print()
    print("\n".join(model.tree_text))
    model.predict_train_data(print_me=True)
