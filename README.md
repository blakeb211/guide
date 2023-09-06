# Minimum Viable Product
1. change parsing file to not change the dataframe but just pass the indexes, split_var etc to node
1. port over tree plotter
1. write predict
1. write test cases for output 
1. baseball dataset
1. consumer expenditure

# Done
1. write fit
1. write split point method G for categorical
1. write split point method M
1. What does M column - missing value flag (codes for missing values)
  - these are only present in some datasets and not present in the 
  final tree for ce2021reg.dsc so we acn save it as a future feature.

# QUESTIONS

1. How are new categories created for missing categoricals
- Create a separate category 'Missing' that they belong to
1. How are missing values in numerical columns handled 
    -  numeric-var <= 10          missing values to the right
    -  numeric-var <= 10   or NA  missing values to the left
    -  numeric-var <= -inf or NA  numeric var equals NA; NA goes left all else goes right

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


1. Parse creates a Settings object
1. Model takes the settings object and builds the model
