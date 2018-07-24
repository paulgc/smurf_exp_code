
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     

from py_stringsimjoin.apply_rf.sim_functions cimport edit_distance
from py_stringsimjoin.apply_rf.inverted_index cimport InvertedIndex             
from py_stringsimjoin.apply_rf.utils cimport build_prefix_index   

cpdef pair[vector[pair[int, int]], vector[double]] ed_join(
                                     vector[vector[int]]& ltokens, 
                                     vector[vector[int]]& rtokens,
                                     int qval,
                                     double threshold,
                                     vector[string]& lstrings,
                                     vector[string]& rstrings, int n_jobs):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs
    cdef vector[vector[double]] output_sim_scores                               
    cdef vector[pair[int, int]] partitions
    cdef int i, n=rtokens.size(), ncpus=n_jobs, partition_size, start=0, end                                   
    cdef InvertedIndex index
    build_prefix_index(ltokens, qval, threshold, index)                                 
    
    partition_size = <int>(<float> n / <float> ncpus)
    print 'part size : ', partition_size
    for i in range(ncpus):
        end = start + partition_size
        if end > n or i == ncpus - 1:
            end = n
        partitions.push_back(pair[int, int](start, end))
        print start, end
        start = end            
        output_pairs.push_back(vector[pair[int, int]]())
        output_sim_scores.push_back(vector[double]())                           

    for i in prange(ncpus, nogil=True):    
        ed_join_part(partitions[i], ltokens, rtokens, qval, threshold, index, 
                     lstrings, rstrings, output_pairs[i], output_sim_scores[i])

    cdef pair[vector[pair[int, int]], vector[double]] output                    
                                                                                
    for i in xrange(ncpus):                                                     
        output.first.insert(output.first.end(),                                 
                            output_pairs[i].begin(), output_pairs[i].end())     
        output.second.insert(output.second.end(),                               
                             output_sim_scores[i].begin(),                      
                             output_sim_scores[i].end())                        

    return output                                                               
                                                                                

cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef void ed_join_part(pair[int, int] partition, 
                       vector[vector[int]]& ltokens, 
                       vector[vector[int]]& rtokens, 
                       int qval, double threshold, InvertedIndex& index,
                       vector[string]& lstrings, vector[string]& rstrings, 
                       vector[pair[int, int]]& output_pairs,
                       vector[double]& output_sim_scores) nogil:    
    cdef oset[int] candidates                                      
    cdef vector[int] tokens
    cdef int j=0, m, i, prefix_length, cand                    
    cdef double edit_dist               
 
    for i in range(partition.first, partition.second):
        tokens = rtokens[i]                        
        m = tokens.size()                                                      
        prefix_length = int_min(<int>(qval * threshold + 1), m)                 
                                                                                
        for j in range(prefix_length):                                          
            if index.index.find(tokens[j]) == index.index.end():                
                continue
            for cand in index.index[tokens[j]]:                                             
                candidates.insert(cand)               

##        print i, candidate_overlap.size()                                      
        for cand in candidates:
            if m - threshold <= index.size_vector[cand] <= m + threshold:
                edit_dist = edit_distance(lstrings[cand], rstrings[i])                                         
                if edit_dist <= threshold:                                       
                    output_pairs.push_back(pair[int, int](cand, i))     
                    output_sim_scores.push_back(edit_dist)                          

        candidates.clear()

