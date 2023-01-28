The code is in the format of .ipynb Jupyter Notebook. It could be run from terminal by the following ways.

nbconvert allows you to run notebooks with the --execute flag:
jupyter nbconvert --execute <notebook>
  
If you want to run a notebook and produce a new notebook, you can add --to notebook:
jupyter nbconvert --execute --to notebook <notebook>
  
Or if you want to replace the existing notebook with the new output:
jupyter nbconvert --execute --to notebook --inplace <notebook>

Since that's a really long command, you can use an alias:
alias nbx="jupyter nbconvert --execute --to notebook"
nbx [--inplace] <notebook>
