# Tree model compatible with GUIDE (Loh et al)
1. Handles numeric and categoric variables with no missing values
1. Currently focusing on compatibility with the 2002 paper (see the docs folder)
1. Original author's site: https://pages.stat.wisc.edu/~loh/guide.html 

# How to run
1. Recommended setup 
 ```
 python3.10 -m venv myenv  # need 3.10 or newer for match statements
 source ./myenv/bin/activate 
 pip install -r requirements.txt
 ```
1. The Makefile is very simple; you can enter the commands manually if you look at it 
1. Normal operation and development: run *iterate.py* by typing `make` from this directory
1. Run test suite with `make test` from this directory

# Holistic Overview
1. *parse.py* creates a Settings object from the .in, .dsc, and .txt files
1. The Model class in *node.py* takes the settings object and builds the model
1. Each directory in the tests folder is a set of GUIDE files input, description, data, output, etc.

