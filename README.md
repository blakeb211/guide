# Minimum Viable Product
1. write predict to generate node file
1. write test that can compare node files
1. match reference for unweighted greedy split of categoricals with no missing values 

# Done
1. match reference with greedy split point, unweighted numeric variables only, no interaction tests, no missing values, no bootstrap collection for piecewise constant

# WISHLIST
1. Categoricals
1. Weights
1. Pruning
1. Missing values
1. Match 1-df chi squared value at root node (we are close but not exact match)

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

# Holistic Overview
1. run tests with `make test` from this directory
1. Parse creates a Settings object
1. Model takes the settings object and builds the model
1. Selecting split variables 

 numerical val          |  0   0.25% | 0.25 to 0.50 | 0.50 to 0.75 | 0.75 - 1.0 |
                   pos
                   neg

 categorical val
                        |   cat1     |   cat2       |    cat3      |   NA       | ... etc
                   pos
                   neg
