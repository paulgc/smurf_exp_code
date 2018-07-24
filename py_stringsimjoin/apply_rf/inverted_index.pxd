
from libcpp.vector cimport vector                                               
from libcpp.map cimport map as omap                                             

cdef extern from "inverted_index.h" nogil:                                      
    cdef cppclass InvertedIndex nogil:                                          
        InvertedIndex()                                                         
        InvertedIndex(omap[int, vector[int]]&, vector[int]&)                    
        void set_fields(omap[int, vector[int]]&, vector[int]&)                  
        omap[int, vector[int]] index                                            
        vector[int] size_vector                
