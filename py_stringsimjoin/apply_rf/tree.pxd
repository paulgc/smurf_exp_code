
from libcpp.vector cimport vector                                               

from py_stringsimjoin.apply_rf.rule cimport Rule                                


cdef extern from "tree.h" nogil:                                                
    cdef cppclass Tree nogil:                                                   
        Tree()                                                                  
        Tree(vector[Rule]&)
        void set_tree_id(int)
        int tree_id                                                     
        vector[Rule] rules  
