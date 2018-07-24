
from libcpp cimport bool                                                        
from libcpp.vector cimport vector                                               


cdef extern from "coverage.h" nogil:                                            
    cdef cppclass Coverage nogil:                                               
        Coverage()                                                              
        Coverage(vector[bool]&)                                                 
        int and_sum(const Coverage&)                                            
        void or_coverage(const Coverage&)                                       
        void and_coverage(const Coverage&)                                      
        void reset()
        int sum()
        int count, size    
