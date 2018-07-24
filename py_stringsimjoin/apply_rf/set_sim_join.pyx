
from cython.parallel import prange                                              

from libc.math cimport ceil, floor, round, sqrt, trunc
from libcpp.vector cimport vector                                               
from libcpp.set cimport set as oset                                             
from libcpp.string cimport string                                               
from libcpp cimport bool                                                        
from libcpp.map cimport map as omap                                             
from libcpp.pair cimport pair     

from py_stringsimjoin.apply_rf.sim_functions cimport cosine, dice, jaccard
from py_stringsimjoin.apply_rf.utils cimport int_min, int_max

cdef double mytrunc(double d) nogil:
    return (trunc(d * 10000) / 10000)

cdef int get_prefix_length(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE
        return <int>(num_tokens - ceil(threshold * threshold * num_tokens) + 1.0)    
    elif sim_type == 1: # DICE
        return <int>(num_tokens - ceil((threshold / (2 - threshold)) * num_tokens) + 1.0)  
    elif sim_type == 2: # JACCARD:
        return <int>(num_tokens - ceil(threshold * num_tokens) + 1.0)

cdef int get_size_lower_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>ceil(threshold * threshold * num_tokens)          
    elif sim_type == 1: # DICE                              
        return <int>ceil((threshold / (2 - threshold)) * num_tokens)                      
    elif sim_type == 2: # JACCARD:                                              
        return <int>ceil(threshold * num_tokens)

cdef int get_size_upper_bound(int& num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>floor(num_tokens / (threshold * threshold))       
    elif sim_type == 1: # DICE                                                  
        return <int>floor(((2 - threshold) / threshold) * num_tokens)   
    elif sim_type == 2: # JACCARD:                                              
        return <int>floor(num_tokens / threshold)       

cdef int get_overlap_threshold(int& l_num_tokens, int& r_num_tokens, int& sim_type, double& threshold) nogil:
    if sim_type == 0: # COSINE                                                  
        return <int>ceil(threshold * sqrt(l_num_tokens * r_num_tokens))    
    elif sim_type == 1: # DICE                                         
        return <int>ceil((threshold / 2) * (l_num_tokens + r_num_tokens))           
    elif sim_type == 2: # JACCARD:                                              
        return <int>ceil((threshold / (1 + threshold)) * (l_num_tokens + r_num_tokens)) 

ctypedef double (*fnptr)(const vector[int]&, const vector[int]&) nogil

cdef fnptr get_sim_function(int& sim_type) nogil:
    if sim_type == 0: # COSINE                                                  
        return cosine
    elif sim_type == 1: # DICE                                                  
        return dice
    elif sim_type == 2: # JACCARD:                                              
        return jaccard

cpdef pair[vector[pair[int, int]], vector[double]] set_sim_join(vector[vector[int]]& ltokens, 
                                          vector[vector[int]]& rtokens,
                                          int sim_type,
                                          double threshold, int n_jobs):                                           
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs
    cdef vector[vector[double]] output_sim_scores
    cdef vector[pair[int, int]] partitions
    cdef int i, n=rtokens.size(), ncpus=n_jobs, partition_size, start=0, end                                   
    cdef PositionIndex index
    build_index(ltokens, sim_type, threshold, index)                                 

    cdef pair[vector[pair[int, int]], vector[double]] output
 
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

    print 'join started'
    for i in prange(ncpus, nogil=True):    
        set_sim_join_part(partitions[i], ltokens, rtokens, sim_type, threshold, 
                          index, output_pairs[i], output_sim_scores[i])

    print 'join finished'

    for i in xrange(ncpus):
        output.first.insert(output.first.end(), 
                            output_pairs[i].begin(), output_pairs[i].end())
        output.second.insert(output.second.end(), 
                             output_sim_scores[i].begin(), 
                             output_sim_scores[i].end())    
    print 'f2'
    return output
#    return pair[vector[pair[int, int]], vector[double]](final_output_pairs,
#                                                        final_output_sim_scores)

cpdef void set_sim_join1(vector[vector[int]]& ltokens,
                                          vector[vector[int]]& rtokens,         
                                          int sim_type,                         
                                          double threshold):                    
    print 'l size. : ', ltokens.size(), ' , r size : ', rtokens.size()          
    cdef vector[vector[pair[int, int]]] output_pairs                            
    cdef vector[vector[double]] output_sim_scores                               
    cdef vector[double] final_output_sim_scores                                 
    cdef vector[pair[int, int]] partitions, final_output_pairs                  
    cdef int i, n=rtokens.size(), ncpus=4, partition_size, start=0, end         
    cdef PositionIndex index                                                    
    build_index(ltokens, sim_type, threshold, index)                            
                                                                                
    cdef pair[vector[pair[int, int]], vector[double]] output                    
                                                                                
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
                                                                                
    print 'join started'                                                        
    for i in prange(ncpus, nogil=True):                                         
        set_sim_join_part1(partitions[i], ltokens, rtokens, sim_type, threshold, 
                          index)         
                                                                                
    print 'join finished'                                                       
                                                                                
    for i in xrange(ncpus):                                                     
        output.first.insert(output.first.end(),                                 
                            output_pairs[i].begin(), output_pairs[i].end())     
        output.second.insert(output.second.end(),                               
                             output_sim_scores[i].begin(),                      
                             output_sim_scores[i].end())                        
    print 'f2'                                                                  
#    return output                                            

cdef void set_sim_join_part1(pair[int, int]& partition,  
                            vector[vector[int]]& ltokens,                       
                            vector[vector[int]]& rtokens,                       
                            int& sim_type,                                       
                            double& threshold, PositionIndex& index) nogil:           
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache              
    cdef vector[pair[int, int]] candidates                                      
    cdef vector[int] tokens                                                     
    cdef pair[int, int] cand, entry                                             
    cdef int k=0, j=0, m, i, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound                           
    cdef double sim_score                                                       
#    cdef fnptr sim_fn                                                           
#    sim_fn = get_sim_function(sim_type)                                         
                                                                                
    for i in range(partition.first, partition.second):                          

        tokens = rtokens[i]                                                     
        m = tokens.size()                                                       
        prefix_length = get_prefix_length(m, sim_type, threshold)               
        size_lower_bound = get_size_lower_bound(m, sim_type, threshold)         
        size_upper_bound = get_size_upper_bound(m, sim_type, threshold)         
#        print i, 'p1'                                                                        
        for size in range(size_lower_bound, size_upper_bound + 1):              
            overlap_threshold_cache[size] = get_overlap_threshold(size, m, sim_type, threshold)
#        print i, 'p2'                                                                        
        
        for j in range(prefix_length):                                          
            if index.index.find(tokens[j]) == index.index.end():
                continue

            candidates = index.index[tokens[j]]                                 
            
            for cand in candidates:                                             
                current_overlap = candidate_overlap[cand.first]                 
               
                if current_overlap != -1:                                       
                    cand_num_tokens = index.size_vector[cand.first]             
                                                                                
                    # only consider candidates satisfying the size filter       
                    # condition.                                                
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound: 
                                                                                
                        if m - j <= cand_num_tokens - cand.second:              
                            overlap_upper_bound = m - j                         
                        else:                                                   
                            overlap_upper_bound = cand_num_tokens - cand.second 
                                                                                
                        # only consider candidates for which the overlap upper  
                        # bound is at least the required overlap.               
                        if (current_overlap + overlap_upper_bound >=            
                                overlap_threshold_cache[cand_num_tokens]):      
                            candidate_overlap[cand.first] = current_overlap + 1 
                        else:                                                   
                            candidate_overlap[cand.first] = -1                  
           
#        print i, 'p3'                                                          
#        print i, candidate_overlap.size()                                      
        for entry in candidate_overlap:                                         
            if entry.second > 0:                                                
#                sim_score = sim_fn(ltokens[entry.first], tokens)                
                #print ltokens[entry.first], rtokens[i], entry.second, sim_score
#                if sim_score > threshold:                                       
                k += 1
#                    output_pairs.push_back(pair[int, int](entry.first, i))      
#                    output_sim_scores.push_back(sim_score)                      
#        print i, 'p4'                                                          
        candidate_overlap.clear()                                               
        overlap_threshold_cache.clear()  

cdef void set_sim_join_part(pair[int, int] partition, 
                            vector[vector[int]]& ltokens, 
                            vector[vector[int]]& rtokens, 
                            int sim_type, 
                            double threshold, PositionIndex& index, 
                            vector[pair[int, int]]& output_pairs,
                            vector[double]& output_sim_scores) nogil:              
    cdef omap[int, int] candidate_overlap, overlap_threshold_cache              
    cdef vector[pair[int, int]] candidates                                      
    cdef vector[int] tokens
    cdef pair[int, int] cand, entry                                             
    cdef int k=0, j=0, m, i, prefix_length, cand_num_tokens, current_overlap, overlap_upper_bound
    cdef int size, size_lower_bound, size_upper_bound                       
    cdef double sim_score               
    cdef fnptr sim_fn
    sim_fn = get_sim_function(sim_type)
 
    for i in range(partition.first, partition.second):
        tokens = rtokens[i]                        
        m = tokens.size()                                                      
        prefix_length = get_prefix_length(m, sim_type, threshold)                    
        size_lower_bound = int_max(get_size_lower_bound(m, sim_type, threshold),
                                   index.min_len)                             
        size_upper_bound = int_min(get_size_upper_bound(m, sim_type, threshold),
                                   index.max_len)                            
#        print i, 'p1'                                                                        
        for size in range(size_lower_bound, size_upper_bound + 1):              
            overlap_threshold_cache[size] = get_overlap_threshold(size, m, sim_type, threshold)
#        print i, 'p2'                                                                        
        for j in range(prefix_length):                                          
            if index.index.find(tokens[j]) == index.index.end():                
                continue  
            candidates = index.index[tokens[j]]                                 
            for cand in candidates:                                             
                current_overlap = candidate_overlap[cand.first]                 
                if current_overlap != -1:                                       
                    cand_num_tokens = index.size_vector[cand.first]             
                                                                                
                    # only consider candidates satisfying the size filter       
                    # condition.                                                
                    if size_lower_bound <= cand_num_tokens <= size_upper_bound: 
                                                                                
                        if m - j <= cand_num_tokens - cand.second:              
                            overlap_upper_bound = m - j                         
                        else:                                                   
                            overlap_upper_bound = cand_num_tokens - cand.second 
                                                                                
                        # only consider candidates for which the overlap upper  
                        # bound is at least the required overlap.               
                        if (current_overlap + overlap_upper_bound >=            
                                overlap_threshold_cache[cand_num_tokens]):      
                            candidate_overlap[cand.first] = current_overlap + 1 
                        else:                                                   
                            candidate_overlap[cand.first] = -1                  
#        print i, 'p3'
#        print i, candidate_overlap.size()                                      
        for entry in candidate_overlap:                                         
            if entry.second > 0:                                                
                sim_score = sim_fn(ltokens[entry.first], tokens)           
                #print ltokens[entry.first], rtokens[i], entry.second, sim_score
                if sim_score > threshold:                                       
                    output_pairs.push_back(pair[int, int](entry.first, i))
                    output_sim_scores.push_back(sim_score)     
#        print i, 'p4'
        candidate_overlap.clear()                                               
        overlap_threshold_cache.clear()    

cdef void build_index(vector[vector[int]]& token_vectors, int& sim_type, double& threshold, PositionIndex &pos_index):
    cdef vector[int] tokens, size_vector                                                 
    cdef int prefix_length, token, i, j, m, n=token_vectors.size(), min_len=100000, max_len=0
    cdef omap[int, vector[pair[int, int]]] index                                 
    for i in range(n):                                                      
        tokens = token_vectors[i]                                           
        m = tokens.size()                                                   
        size_vector.push_back(m)                                       
        prefix_length = get_prefix_length(m, sim_type, threshold)                  
        for j in range(prefix_length):                                      
            index[tokens[j]].push_back(pair[int, int](i, j))           
        if m > max_len:                                                     
            max_len = m                                                     
        if m < min_len:                                                     
            min_len = m
    pos_index.set_fields(index, size_vector, min_len, max_len, threshold)

cdef extern from "position_index.h" nogil:
    cdef cppclass PositionIndex nogil:
        PositionIndex()
        PositionIndex(omap[int, vector[pair[int, int]]]&, vector[int]&, int&, int&, double&)
        void set_fields(omap[int, vector[pair[int, int]]]&, vector[int]&, int&, int&, double&)
        omap[int, vector[pair[int, int]]] index
        int min_len, max_len
        vector[int] size_vector
        double threshold
