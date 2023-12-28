# Current feature goal
1. Improve output match with the reference, adding necessary features until variable importance can be calculated on arbitrary tabular datasets and used in production for data reduction tasks.

# Current supported features
1. Handles numeric and categoric variables with no missing values

# Todo 
1. Decide what order to attack the wishlist
2. Put goals at top of TODO
# Completed 
1. Add one paragraph summary of what is Guide and what has been done so far
1. Add CONTRIBUTORS.md file
1. Open the repo
1. Add Github Action CI/CD
1. Add predict for arbitrary dataframes
1. MyLint and MyPy + code cleanup 
1. Root cause and fixing of unbiased selection test discrepancies
1. Write test for unbiased selection from the 2002 paper
1. Wire up Interaction tests
1. Put in fast categoricals
1. Switched goal from "exact match to GUIDE v41.2 to 'GUIDE-Compatible implementation"
1. Match reference for unweighted greedy split of categoricals with no missing values 
1. Match reference with greedy split point, unweighted numeric variables only, no interaction tests, no missing values, no bootstrap collection for piecewise constant
1. Write test to compare the tree structure outputted by GUIDE
1. Write predict_train_data to generate node file (runs predict on the training data)
1. Write test that can compare node files

# Wishlist
1. Add node count comparison to testing output. I believe it differs a fair bit from the reference for some tests. Want to reduce the gap.
1. Add fraction of variance explained by model printout
1. Add another test for the interaction test part of the code. 
1. Add Pruning with some common options like 0-SE or 1-SE
1. Regression for piecewise linear 
  1. Bootstrap correction 
  1. Will have to implement more variable roles
  1. Bootstrap bias correction is for linear models 
1. Classification
1. Priors / Misclassification costs, where does it fit in?
1. Missing values
1. Weights
1. Variable importance
1. Convert to internal ndarrays instead of dataframes 
1. Reduce dependencies

# Notes 
## Variable roles
1. Planned to include (only n and c in now) n, s, c, b
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

