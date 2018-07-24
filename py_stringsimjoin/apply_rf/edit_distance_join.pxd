
from libcpp.vector cimport vector    
from libcpp.string cimport string                                           
from libcpp.pair cimport pair                                                   

cpdef pair[vector[pair[int, int]], vector[double]] ed_join(
                                     vector[vector[int]]& ltokens, 
                                     vector[vector[int]]& rtokens,
                                     int qval, 
                                     double threshold,
                                     vector[string]& lstrings, 
                                     vector[string]& rstrings, int n_jobs)
