# How to run
1. Recommended setup 
 ```
 python3.10 -m venv myenv  # need 3.10 or newer for match statements
 source ./myenv/bin/activate 
 pip install -r requirements.txt
 ```
1. The Makefile is very simple; you can enter the commands manually if you look at it 
1. Normal operation and development: run *iterate.py* by typing `make` from this directory
1. Run test suite with `make test` from this directory. 

# Holistic Overview
1. *parse.py* creates a Settings object from the .in, .dsc, and .txt files
1. The Model class in *node.py* takes the settings object and builds the model
1. Each directory in the tests folder is a set of GUIDE files input, description, data, output, etc.

# Todo 
1. Put in fast categoricals
1. Put in Interaction tests
1. Write test for interaction tests from the 2002 paper
1. Clean up parse 
1. MyLint and MyPy to clean up the code
1. Open the repo

# Completed 
1. switched goal from "exact match to GUIDE v41.2 to 'GUIDE-Compatible implementation"
1. match reference for unweighted greedy split of categoricals with no missing values 
1. match reference with greedy split point, unweighted numeric variables only, no interaction tests, no missing values, no bootstrap collection for piecewise constant
1. write predict to generate node file
1. write test that can compare node files

# Wishlist
1. Pruning
1. Regression for piecewise linear 
  1. Bootstrap correction 
  1. Will have to implement more variable roles
  1. Bootstrap bias correction is for linear models 
1. Classification
1. Priors / Misclassification costs, where does it fit in?
1. Missing values
1. Weights
1. Match 1-df chi squared value at root node (we are close but not exact match)
1. Reduce dependencies
1. Increase speed

# Notes 
## Variable roles
1. n, s, c, b
1. For regression, variable can be used for splitting, node modeling, or both
1. Categorical predictions not used for node modeling; if want that must dummy encode
1. All 'n' vars converted to 's' for piecewise constant models (no regressor vars)

## Missing values
1. How are new categories created for missing categoricals
- Create a separate category 'Missing' that they belong to
1. How are missing values in numerical columns handled 
    -  numeric-var <= 10          missing values to the right
    -  numeric-var <= 10   or NA  missing values to the left
    -  numeric-var <= -inf or NA  numeric var equals NA; NA goes left all else goes right
1. What does M column - missing value flag (codes for missing values)
  - these are only present in some datasets. Save it as a future feature.

Table: Variable role meanings
| Abbreviation | Meaning                                                   |
|--------------|-----------------------------------------------------------|
| d            | Dependent variable                                       |
| b            | Split and fit categorical variable using indicator variables |
| c            | Split-only categorical variable                         |
| i            | Fit-only categorical variable via indicators            |
| s            | Split-only numerical variable                           |
| n            | Split and fit numerical variable                        |
| f            | Fit-only numerical variable                             |
| m            | Missing-value flag variable                              |
| p            | Periodic variable                                       |
| w            | Weight                                                   |

