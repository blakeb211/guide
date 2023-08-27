1. read manual
1. TO RUN BINARY:
  - interactive prompt creates input file
  - ./guide < input.txt

1. try to match ce2021reg output 
1. write minimum version of algorithm
1. write testing against reference output

# QUESTIONS
1. What does M column - missing value flag (codes for missing values)

1. How are new categories created for missing categoricals
- Create a separate category that they all belong to

1. How are missing values in numerical columns handled

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

