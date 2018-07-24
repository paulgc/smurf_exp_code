
from libcpp.vector cimport vector                                               

from py_stringsimjoin.apply_rf.predicatecpp cimport Predicatecpp                


cdef extern from "rule.h" nogil:                                                
    cdef cppclass Rule nogil:                                                   
        Rule()                                                                  
        Rule(vector[Predicatecpp]&)                                             
        vector[Predicatecpp] predicates

