# smurf_exp_code

Dependencies: pandas, scikit-learn, pyre2, re2 (C++), six, Cython, joblib, py_stringmatching, Pyprind.

There are two main scripts:
- learn_rf.py : To learn a random forest
- execute_rf.py : To execute a random forest


Before running these scripts, first compile all Cython code in py_stringsimjoin/apply_rf. Specifically, do the following
  - cd py_stringsimjoin/apply_rf
  - python setup.py build_ext --inplace
  
