import pdb
import sys
import pandas as pd
import numpy as np
import subprocess
sys.path.append("../..")
from node import Model
from parse import Settings
import hashlib
# List of filenames
file_list = [f'data-indep{i}.txt' for i in range(0,1000)]

results = {}
results_my_model = {}
# Loop over each filename
for filename in file_list:
    # Task 1: Change the first line of data.dsc to the current filename
    with open("data.dsc", "r") as f:
        lines = f.readlines()
    lines[0] = filename + "\n"
    with open("data.dsc", "w") as f:
        f.writelines(lines)

    # Task 2: Run the bash command
    command = f"../../guide < cons.in"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Task 3: Find and output the desired line in cons.out
    with open("cons.out", "r") as f:
        lines = f.readlines()
        start_index = None
        for i, line in enumerate(lines):
            if "Top-ranked variables and 1-df chi-squared values at root node" in line:
                start_index = i
                break

        if start_index is not None and start_index + 1 < len(lines):
            output_line = lines[start_index + 1].strip()
            results[output_line[16:]] = results.get(output_line[16:],0) + 1

    _settings = Settings("./","data.dsc",max_depth=2,min_samples_leaf=2,input_file="cons.in")
    _model = Model(_settings)
    _model.fit()
    results_my_model[_model.top_node_best_var] = results_my_model.get(_model.top_node_best_var,0) + 1

print("GUIDE:")
print(results)
print("My model:")
print(results_my_model)
