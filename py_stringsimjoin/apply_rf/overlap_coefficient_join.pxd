
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair                                                   

cpdef pair[vector[pair[int, int]], vector[double]] ov_coeff_join(
                                           vector[vector[int]]& ltokens, 
                                           vector[vector[int]]& rtokens,
                                           double threshold, int n_jobs)
