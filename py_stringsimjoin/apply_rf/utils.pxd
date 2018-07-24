
from libcpp cimport bool
from libcpp.vector cimport vector                                               
from libcpp.string cimport string 

from py_stringsimjoin.apply_rf.inverted_index cimport InvertedIndex             


ctypedef double (*token_simfnptr)(const vector[int]&, const vector[int]&) nogil       
ctypedef double (*str_simfnptr)(const string&, const string&) nogil       

ctypedef double (*simfnptr)(const vector[int]&, const vector[int]&) nogil       
ctypedef double (*simfnptr_str)(vector[string]&, vector[string]&) nogil       
ctypedef bool (*compfnptr)(double, double) nogil                                

cdef int get_sim_type(const string&)
cdef simfnptr get_sim_function(const int) nogil
cdef simfnptr_str get_sim_function_str(const int) nogil                                 

cdef token_simfnptr get_token_sim_function(const int) nogil                                 
cdef str_simfnptr get_str_sim_function(const int) nogil                                 

cdef int get_comp_type(const string&)
cdef compfnptr get_comparison_function(const int) nogil

cdef void build_inverted_index(vector[vector[int]]& token_vectors, InvertedIndex &inv_index)
cdef void build_prefix_index(vector[vector[int]]& token_vectors, int qval, double threshold, InvertedIndex &inv_index)

cdef int int_min(int a, int b) nogil
cdef int int_max(int a, int b) nogil
