# Todo 
1. switch goal from "exact match to GUIDE v41.2 to 'GUIDE-Compatible implementation"

# Done
1. match reference for unweighted greedy split of categoricals with no missing values 
1. match reference with greedy split point, unweighted numeric variables only, no interaction tests, no missing values, no bootstrap collection for piecewise constant
1. write predict to generate node file
1. write test that can compare node files

# WISHLIST
1. Pruning
1. Interaction tests
1. Missing values
1. Weights
1. Regression for piecewise best simple linear model 
1. Classification
1. Bootstrap correction (is this only for categoricals?)
1. Priors / Misclassification costs, where does it fit in?
1. Match 1-df chi squared value at root node (we are close but not exact match)
1. Reduce dependencies
1. Increase speed

## Variable roles
1. For regression, variable can be used for splitting, node modeling, or both

## Missing values
1. How are new categories created for missing categoricals
- Create a separate category 'Missing' that they belong to
1. How are missing values in numerical columns handled 
    -  numeric-var <= 10          missing values to the right
    -  numeric-var <= 10   or NA  missing values to the left
    -  numeric-var <= -inf or NA  numeric var equals NA; NA goes left all else goes right
1. What does M column - missing value flag (codes for missing values)
  - these are only present in some datasets. Save it as a future feature.

## Pruning
1. How is cost complexity calculated for regression? (I have a note what 0-SE is)
  pruning on SSE of residuals

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

# How to run
1. run iterate.py 'make' from this directory
1. run tests with `make test` from this directory
1. the Makefile is very simple; you do not need make if you just look at it

# Holistic Overview
1. Parse creates a Settings object from the .in, .dsc, and .txt files
1. Model takes the settings object and builds the model
1. Testing

