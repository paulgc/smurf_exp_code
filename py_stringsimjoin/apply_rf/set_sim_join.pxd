
from libcpp.vector cimport vector                                               
from libcpp.pair cimport pair                                                   

cpdef pair[vector[pair[int, int]], vector[double]] set_sim_join(vector[vector[int]]& ltokens, 
                                          vector[vector[int]]& rtokens,
                                          int sim_type,
                                          double threshold, int n_jobs)

cpdef void set_sim_join1(vector[vector[int]]& ltokens,
                                          vector[vector[int]]& rtokens,         
                                          int sim_type,                         
                                          double threshold)
