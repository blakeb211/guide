# Tree model compatible with GUIDE (Loh et al)
1. Handles numeric and categoric variables with no missing values
1. I'm still defining what subset of the (massive) program options that I'll support 
1. Original author's site: https://pages.stat.wisc.edu/~loh/guide.html 
1. Tree book that inspired the project: https://bgreenwell.github.io/treebook/

[![Python application](https://github.com/blakeb211/guide/actions/workflows/python-app.yml/badge.svg?branch=master&event=push)](https://github.com/blakeb211/guide/actions/workflows/python-app.yml)

# Usage 
 ```
 python3.10 -m venv myenv  # need 3.10 or newer for match statements
 source ./myenv/bin/activate 
 pip install -r requirements.txt
 ```
1. Normal operation and development: edit and run *iterate.py* (or just type `make`)
1. Run dataset test suite with `make test`
1. Run unbiased selection test (could take 10 min) `make test_unbiased`

# Holistic Overview
1. *parse.py* creates a Settings object from the .in, .dsc, and .txt files
1. The Model class in *node.py* takes the settings object and builds the model
1. Each directory in the tests folder is a set of GUIDE files like input, 
description, data, output, etc.

