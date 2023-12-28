# Tree model compatible with GUIDE (Loh et al)
1. GUIDE is a lesser-known tree algorithm very similar to CART but with improvements like unbiased selection of variables. CART is known to prefer split variables that have a large number of unique values or categories. There are useful features of the program beyond that, but it is the first killer feature. (See docs folder). GUIDE is not an open source program at the time of this writing and is distributed as binaries, which limits users' ability to tweak it. 
1. Original author's site: https://pages.stat.wisc.edu/~loh/guide.html 
1. Tree book that inspired the project: https://bgreenwell.github.io/treebook/
   
[![Python application](https://github.com/blakeb211/guide/actions/workflows/python-app.yml/badge.svg?branch=master&event=push)](https://github.com/blakeb211/guide/actions/workflows/python-app.yml)

# Example output from for data-tiniest2 test 
```
>> Parsed description file <<
**************************************************
Datafile name            : data.txt
Missing value labels     : ['NA']
Number of variables      : 4
Variable types           : {'d': 1, 'n': 3}
**************************************************
Number of rows datafile  : 60
Dependent variable       : target
Model type is            : RegressionType.PIECEWISE_CONSTANT
Dropped missing d rows   : 0
Converted 3 n variables to S variables


  num var_name var_role       min       max levels missing
3   1   target        d                                   
4   2     num1        S       1.0       6.9               
5   3     num2        S         0         9               
6   4     num3        S  0.060194  0.947571               
Number of split variables: 3
Max depth of tree     : 3
Min samples per node  : 6
Interaction tests done: False

Node 1: num1 <= 3.8499999999999996
  Node 2: num2 <= 5.5
    Node 4: target-mean =  6.971429
  Node 2: num2 > 5.5
    Node 5: target-mean = 12.725000
Node 1: num1 > 3.8499999999999996
  Node 3: num2 <= 5.5
    Node 6: target-mean = 12.971429
  Node 3: num2 > 5.5
    Node 7: target-mean = 18.940000
```

# Usage 
 ```
 python3.10 -m venv myenv
 source ./myenv/bin/activate 
 pip install -r requirements.txt
 ```
1. Normal operation and development: *python main.py*
1. Run dataset test suite with `pytest -rA --log-cli-level INFO test_datasets.py` These tests compare this program's output to the GUIDE output (called 'this prog' and 'reference' respectively in the code).
1. Run unbiased selection test. (could take 10 min) `pytest -rA --log-cli-level INFO -vv --tb=long test_unbiased.py`
1. Look at the TODO list and CONTRIBUTING.md to make your contribution [Link to TODO.md](TODO.md)

# Code Overview
1. Each directory in the tests folder is a set of GUIDE files like input, 
description, data, output, etc. 
1. *main.py* shows how to create and fit and model on a set of GUIDE input files. Only certain options are supported. Looking through
the input files in the tests/data-* directory (named *cons.in* for no particular reason) tells what parameters are supported.
1. *parse.py* create a Settings object from the .in, .dsc, and .txt files
1. The Model class in *node.py* takes the settings object and builds the model

# GUIDE binary and datafiles
* The guide executable is in this folder for testing. Example, from folder `./tests/data-new-dataset` you could run `../../guide < cons.in` to generate the GUIDE output files for that folder.

