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
        data_dir="./tests/data-unbiased-selection/",
        dsc_file="data.dsc",
        model=RegressionType.PIECEWISE_CONSTANT,
        out_file="cons.out",
        input_file="cons.in",
        overwrite_data_txt="data-weak22.txt",
    )
    model = Model(settings)
    model.fit()
    print("\n".join(model.tree_text))
    # model.predict_train_data(print_me=True)
    pdb.set_trace()
    test_X = pd.DataFrame([[0.3,0.4,3,3,3]], columns=["X1","X2","X3","X4","X5"],index=[0])
    predictions = model.predict(test_X)
    print(predictions)
