# Todo 
1. Add Github Action CI/CD
1. Open the repo

# Completed 
1. Add predict for arbitrary dataframes
1. MyLint and MyPy + code cleanup 
1. Root cause and fixing of unbiased selection test discrepancies
1. Write test for interaction tests from the 2002 paper
1. Wire up Interaction tests
1. Put in fast categoricals
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

