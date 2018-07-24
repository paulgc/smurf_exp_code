
from cython.parallel import prange                                              

from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     

from py_stringsimjoin.apply_rf.inverted_index cimport InvertedIndex      
from py_stringsimjoin.apply_rf.utils cimport build_inverted_index

cpdef pair[vector[pair[int, int]], vector[double]] ov_coeff_join(vector[vector[int]]& ltokens, 
                                           vector[vector[int]]& rtokens,
                                           double threshold, int n_jobs):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs
    cdef vector[vector[double]] output_sim_scores                               
    cdef vector[pair[int, int]] partitions
    cdef int i, n=rtokens.size(), ncpus=n_jobs, partition_size, start=0, end                                   
    cdef InvertedIndex index
    build_inverted_index(ltokens, index)                                 
    
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
        ov_coeff_join_part(partitions[i], ltokens, rtokens, threshold, index, 
                           output_pairs[i], output_sim_scores[i])

    cdef pair[vector[pair[int, int]], vector[double]] output                    

    for i in xrange(ncpus):                                                     
        output.first.insert(output.first.end(),                                 
                            output_pairs[i].begin(), output_pairs[i].end())     
        output.second.insert(output.second.end(),                               
                             output_sim_scores[i].begin(),                      
                             output_sim_scores[i].end())    

    return output                                                               


cdef inline int int_min(int a, int b) nogil: return a if a <= b else b

cdef void ov_coeff_join_part(pair[int, int] partition, 
                             vector[vector[int]]& ltokens, 
                             vector[vector[int]]& rtokens, 
                             double threshold, InvertedIndex& index, 
                             vector[pair[int, int]]& output_pairs,
                             vector[double]& output_sim_scores) nogil:              
    cdef omap[int, int] candidate_overlap              
    cdef vector[int] candidates                                      
    cdef vector[int] tokens
    cdef pair[int, int] entry                                             
    cdef int j=0, m, i, cand                    
    cdef double sim_score               
 
    for i in range(partition.first, partition.second):
        tokens = rtokens[i]                        
        m = tokens.size()                                                      
                                                                                
        for j in range(m):                                          
            if index.index.find(tokens[j]) == index.index.end():                
                continue   
            candidates = index.index[tokens[j]]                                 
            for cand in candidates:                                             
                candidate_overlap[cand] += 1                

###        print i, candidate_overlap.size()                                      
        for entry in candidate_overlap:                                         
            sim_score = <double>entry.second / <double>int_min(m, index.size_vector[entry.first])           
            #print ltokens[entry.first], rtokens[i], entry.second, sim_score
            if sim_score > threshold:                                       
                output_pairs.push_back(pair[int, int](entry.first, i))     
                output_sim_scores.push_back(sim_score)                      

        candidate_overlap.clear()                                               

